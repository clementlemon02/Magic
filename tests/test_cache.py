"""Permission-aware cache. A semantic cache in a permissioned system is a
disclosure hole unless partitioned correctly, so these are security tests.
"""

from datetime import timedelta

import pytest

from src.cache import PermissionAwareCache
from src.graph.state import AskerResponse, Citation, UserContext

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)
MARCUS = UserContext(id=2, role="compliance", dept="compliance", clearance_level=1)
# Same role and department as Alex, so their ACL tags are identical.
JAMIE = UserContext(id=3, role="support", dept="support", clearance_level=0)


class FakeEmbeddings:
    """Bag of words over a tiny vocabulary — deterministic, and paraphrases match."""

    VOCAB = ["refund", "chargeback", "review", "days", "aml", "escalation", "window", "outage"]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        vector = [1.0 if word in lowered else 0.0 for word in self.VOCAB]
        return vector if any(vector) else [1.0] + [0.0] * (len(self.VOCAB) - 1)


def _grants(mapping, epoch=None):
    """Fake the permissions table, {user_id: [(platform, ref), ...]}, and corpus_state.

    `epoch` is a one-item list so a test can bump it the way a sync does.
    """

    def execute(sql, params):
        if "corpus_state" in sql:
            return [(epoch[0],)] if epoch else []
        return list(mapping.get(params["user_id"], []))

    return execute


def _cache(mapping, epoch=None, **options):
    return PermissionAwareCache(
        embeddings=FakeEmbeddings(), execute=_grants(mapping, epoch), **options
    )


ANSWER = AskerResponse(
    text="Support must review refund requests within five business days.",
    citations=[
        Citation(
            document_id=1,
            title="Refund Policy",
            source_platform="confluence",
            source_ref="SUPPORT/refund-policy",
        )
    ],
)


def test_same_caller_same_question_hits():
    cache = _cache({1: [("confluence", "SUPPORT/refund-policy")]})
    cache.put("refund review days", ALEX, ANSWER)
    assert cache.get("refund review days", ALEX) is not None


def test_a_paraphrase_hits():
    """Differently worded, same terms — this is what makes it semantic rather than exact."""
    cache = _cache({1: [("confluence", "SUPPORT/refund-policy")]})
    cache.put("How long is the refund review?", ALEX, ANSWER)
    assert cache.get("Tell me about the refund review, please", ALEX) is not None


def test_a_different_caller_never_reads_a_cached_answer():
    """The whole point. Marcus's answer must not reach Alex through the cache."""
    cache = _cache({
        1: [("confluence", "SUPPORT/refund-policy")],
        2: [("confluence", "SUPPORT/refund-policy"), ("confluence", "COMPLIANCE/aml-escalation")],
    })
    cache.put("aml escalation", MARCUS, AskerResponse(text="Tier 2 review above SGD 9,500."))
    assert cache.get("aml escalation", ALEX) is None


def test_callers_with_identical_access_share_entries():
    """Partitioning must not degenerate into a per-user cache."""
    grants = [("confluence", "SUPPORT/refund-policy")]
    cache = _cache({1: grants, 3: grants})
    cache.put("refund review", ALEX, ANSWER)
    assert cache.get("refund review", JAMIE) is not None


def test_revoking_a_grant_invalidates_the_entry():
    """Scenario 4, for free: the fingerprint changes, so the entry stops matching."""
    live = {2: [("confluence", "COMPLIANCE/aml-escalation")]}
    cache = _cache(live)
    cache.put("aml escalation", MARCUS, AskerResponse(text="Tier 2 review."))
    assert cache.get("aml escalation", MARCUS) is not None

    live[2] = []  # the admin revoke endpoint sets revoked_at
    assert cache.get("aml escalation", MARCUS) is None


def test_granting_new_access_also_invalidates():
    """A caller who gains access must not keep seeing the narrower cached answer."""
    live = {1: [("confluence", "SUPPORT/refund-policy")]}
    cache = _cache(live)
    cache.put("refund review", ALEX, ANSWER)
    live[1] = live[1] + [("confluence", "COMPLIANCE/aml-escalation")]
    assert cache.get("refund review", ALEX) is None


def test_an_unrelated_question_misses():
    cache = _cache({1: [("confluence", "SUPPORT/refund-policy")]})
    cache.put("refund review days", ALEX, ANSWER)
    assert cache.get("outage", ALEX) is None


def test_fingerprint_covers_grants_not_just_role():
    """Two callers can share a role and still differ by one explicit grant."""
    cache = _cache({
        1: [("confluence", "SUPPORT/refund-policy")],
        3: [("confluence", "SUPPORT/refund-policy"), ("drive", "file-secret")],
    })
    assert cache.fingerprint(ALEX) != cache.fingerprint(JAMIE)


@pytest.mark.parametrize("threshold", [0.93, 0.99])
def test_threshold_is_respected(threshold):
    cache = PermissionAwareCache(
        embeddings=FakeEmbeddings(), execute=_grants({1: []}), similarity_threshold=threshold
    )
    cache.put("refund review days", ALEX, ANSWER)
    # Shares two of three terms, so cosine is about 0.82 — below either floor.
    assert cache.get("refund review outage", ALEX) is None


# --- freshness: what a sync (src/ingestion/sync.py) does to cached answers ---------------

def test_a_sync_invalidates_answers_cached_before_it():
    """The demo failure, if this is wrong: edit a document, ask again, get the OLD answer
    from the cache. The sync bumps corpus_state.epoch, so nothing cached earlier matches."""
    epoch = [0]
    cache = _cache({1: [("confluence", "SUPPORT/refund-policy")]}, epoch)
    cache.put("refund review days", ALEX, ANSWER)
    assert cache.get("refund review days", ALEX) is not None

    epoch[0] = 1  # a sync changed a document
    assert cache.get("refund review days", ALEX) is None


def test_an_unchanged_corpus_keeps_its_entries():
    """A sync that finds nothing to do must not bump the epoch, or every scheduled run
    would empty the cache and the latency win with it. Nothing here bumps it."""
    epoch = [3]
    cache = _cache({1: [("confluence", "SUPPORT/refund-policy")]}, epoch)
    cache.put("refund review days", ALEX, ANSWER)
    assert cache.get("refund review days", ALEX) is not None


def test_an_answer_is_filed_under_the_state_it_was_computed_in():
    """A request straddling a sync computed its answer from the OLD corpus. Filed under a
    fingerprint read after the sync, it would sit under the NEW epoch and be served as
    current until it expired. `/query` pins the fingerprint before the graph runs."""
    epoch = [0]
    cache = _cache({1: [("confluence", "SUPPORT/refund-policy")]}, epoch)
    before = cache.fingerprint(ALEX)

    epoch[0] = 1  # the sync lands while the graph is still running
    cache.put("refund review days", ALEX, ANSWER, fingerprint=before)

    assert cache.get("refund review days", ALEX) is None


def test_an_entry_expires():
    """The backstop for any change the epoch cannot see."""
    cache = _cache({1: [("confluence", "SUPPORT/refund-policy")]}, ttl_seconds=3600)
    cache.put("refund review days", ALEX, ANSWER)
    assert cache.get("refund review days", ALEX) is not None

    cache._entries[0].stored_at -= timedelta(hours=2)
    assert cache.get("refund review days", ALEX) is None


def test_no_ttl_means_no_expiry():
    cache = _cache({1: [("confluence", "SUPPORT/refund-policy")]})
    cache.put("refund review days", ALEX, ANSWER)
    cache._entries[0].stored_at -= timedelta(days=30)
    assert cache.get("refund review days", ALEX) is not None
