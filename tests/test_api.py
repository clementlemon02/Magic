"""POST /query. The trust boundary: identity is resolved server-side, and the
response cannot carry the reason for a refusal.
"""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import get_settings
from src.graph.state import (
    GENERIC_REFUSAL,
    AuditEvent,
    Citation,
    PermConflict,
    UserContext,
    VerificationResult,
)
from tests.helpers import as_user


@pytest.fixture(autouse=True)
def pinned_settings(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MAX_HOPS", "3")
    monkeypatch.setenv("VERIFIER_CONFIDENCE_THRESHOLD", "0.6")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)


def _citation() -> Citation:
    return Citation(
        document_id=1,
        title="Customer Refund & Chargeback Policy",
        source_platform="confluence",
        source_ref="SUPPORT/refund-policy",
    )


def _nodes(**overrides):
    base = {
        "router": lambda s: {"route": "rag"},
        "retrieval": lambda s: {"hop_count": s.get("hop_count", 0) + 1},
        "sql_tool": lambda s: {"sql_result": None},
        "clarification": lambda s: {"clarification_question": "Which ticket?"},
        "synthesizer": lambda s: {"draft_answer": "45 days.", "citations": [_citation()]},
        "verifier": lambda s: {
            "verification": VerificationResult(grounded=True, unsupported=[], confidence=0.9)
        },
        "escalation": lambda s: {"escalated": True, "explanation": "matched a Compliance-only page"},
        "audit": lambda s: {
            "audit_events": [
                AuditEvent(event_type="final_answer", payload={}, occurred_at=datetime.now(UTC))
            ]
        },
    }
    base.update(overrides)
    return base


def _client(**overrides) -> TestClient:
    return TestClient(create_app(nodes=_nodes(**overrides), user_loader=lambda uid: ALEX))


def test_answers_a_permitted_question_with_citations():
    r = _client().post("/query", json={"query": "How long to contest a chargeback?"}, headers=as_user(1))
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "45 days."
    assert body["citations"][0]["source_ref"] == "SUPPORT/refund-policy"


def test_refusal_body_contains_no_reason():
    """Scenario 3 at the HTTP boundary — the wire format itself must not leak."""
    r = _client(
        retrieval=lambda s: {
            "hop_count": 1,
            "permission_conflicts": [
                PermConflict(
                    document_id=2,
                    source_platform="confluence",
                    source_ref="COMPLIANCE/aml-escalation",
                    sensitivity="restricted",
                    score_margin=0.2,
                )
            ],
        }
    ).post("/query", json={"query": "What triggers an AML escalation?"}, headers=as_user(1))

    assert r.status_code == 200
    body = r.json()
    assert body["text"] == GENERIC_REFUSAL
    assert body["citations"] == []
    # A tripwire on the shape, so a field added to AskerResponse has to be considered
    # here before it can ride out on a refusal. `trace` joined it when the UI grew a
    # pipeline view: empty on every refusal, because a trace reconstructs the cause
    # the wording withholds — a permission conflict stops at retrieval on hop 1, an
    # unsupported answer grinds through three.
    assert set(body) == {"text", "citations", "trace"}, f"unexpected keys: {set(body)}"
    assert body["trace"] == []

    serialised = r.text.lower()
    for leak in ("explanation", "aml", "compliance", "restricted", "conflict"):
        assert leak not in serialised


def test_caller_cannot_supply_its_own_role_or_clearance():
    """Accepting any of these from the body would walk straight past the §1 filter.

    `user_id` is on this list now too: it was a real field until the caller came from
    a signed token, and it was never checked against anything."""
    r = _client().post(
        "/query",
        json={"query": "anything", "user_id": 1, "role": "compliance", "clearance_level": 1},
        headers=as_user(1),
    )
    assert r.status_code == 422


def test_a_token_naming_nobody_is_rejected_without_saying_so():
    """401, not 404. A 404 answered the question "does user 999 exist?", which is a
    free account enumeration for anyone holding any valid signing key."""
    app = create_app(nodes=_nodes(), user_loader=lambda uid: None)
    r = TestClient(app).post("/query", json={"query": "anything"}, headers=as_user(999))
    assert r.status_code == 401
    assert "999" not in r.text and "user" not in r.json()["detail"].lower()


def test_empty_query_is_rejected():
    assert _client().post("/query", json={"query": ""}, headers=as_user(1)).status_code == 422


def test_identity_used_downstream_is_the_loaded_one_not_the_request():
    seen = {}

    def capture(state):
        seen["user"] = state["user"]
        return {"route": "rag"}

    client = TestClient(
        create_app(nodes=_nodes(router=capture), user_loader=lambda uid: ALEX)
    )
    client.post("/query", json={"query": "anything"}, headers=as_user(1))
    assert seen["user"].role == "support"
    assert seen["user"].acl_tags() == ["support", "support", "all-staff"]


# --- startup warm-up -----------------------------------------------------------

@pytest.mark.asyncio
async def test_warm_up_does_nothing_when_it_is_off():
    """conftest.py turns it off for the suite: every create_app would otherwise load
    a 7B model to answer nothing."""
    from src.api.main import _warm
    from src.config import Settings

    await _warm(Settings(_env_file=None, warm_on_startup=False))  # must not raise


@pytest.mark.asyncio
async def test_a_failed_warm_up_is_a_slow_first_answer_not_a_broken_start(monkeypatch):
    """The server is already serving when this runs. A model that will not load is
    a cold first question, never a failed startup."""
    from src.api import main
    from src.config import Settings

    def unreachable():
        raise ConnectionError("ollama is not running")

    monkeypatch.setattr(main, "run_in_threadpool", lambda fn: unreachable())
    await main._warm(Settings(_env_file=None, warm_on_startup=True))  # must not raise


def test_an_answer_carries_its_trace_and_a_refusal_does_not():
    """The pipeline the UI replays. On an answer it is what the graph did; on a
    refusal it is empty, because its shape names the cause the sentence withholds."""
    from src.graph.graph import build_response
    from src.graph.state import AuditEvent

    ran = [
        AuditEvent(event_type="node_transition",
                   payload={"node": "retrieval", "hop": 1, "ms": 48.3},
                   occurred_at=datetime.now(UTC)),
        AuditEvent(event_type="node_transition",
                   payload={"node": "verifier", "hop": 1, "ms": 2361.9},
                   occurred_at=datetime.now(UTC)),
    ]
    answered = build_response(
        {"escalated": False, "final_answer": "45 days.", "citations": [], "audit_events": ran}
    )
    assert [s.node for s in answered.trace] == ["retrieval", "verifier"]
    assert answered.trace[1].ms == 2361.9

    refused = build_response(
        {"escalated": True, "final_answer": "45 days.", "citations": [], "audit_events": ran}
    )
    assert refused.trace == []


def test_a_cache_hit_reports_the_cache_not_the_run_it_replays():
    """The cached response carries the trace of the request that FILLED the cache.
    Passing it through would show seconds of retrieval and synthesis that did not
    happen on this request — a plausible, wrong number next to a 0.1s clock."""
    from src.graph.state import AskerResponse, Stage

    stored = AskerResponse(
        text="45 days.",
        citations=[_citation()],
        trace=[Stage(node="synthesizer", hop=1, ms=2704.4)],
    )

    from src.cache import CacheEntry

    entry = CacheEntry(
        vector=[], fingerprint="f", response=stored,
        stored_at=datetime.now(UTC), route="sql",
    )

    class AlwaysHits:
        def lookup(self, query, user):
            return entry

        def put(self, query, user, response, route=None):
            raise AssertionError("a hit must not re-cache")

    audited = {}
    nodes = _nodes(audit=lambda s: audited.update(s) or {"audit_events": []})
    app = create_app(nodes=nodes, user_loader=lambda uid: ALEX, cache=AlwaysHits())
    body = TestClient(app).post("/query", json={"query": "anything"}, headers=as_user(1)).json()

    assert body["text"] == "45 days."
    assert [s["node"] for s in body["trace"]] == ["cache"]
    assert stored.trace[0].node == "synthesizer", "the cached entry was mutated"
    # The audit records how the answer was ORIGINALLY produced. Without this it gets
    # initial_state's "rag" placeholder, because the Router never ran.
    assert audited["route"] == "sql", "a cached sql answer was audited as rag"


# --- /query rate limiting -------------------------------------------------------

def _limited(limit: int, **overrides):
    from src.api.auth import SlidingWindow

    return TestClient(create_app(
        nodes=_nodes(**overrides),
        user_loader=lambda uid: ALEX,
        query_limit=SlidingWindow(limit=limit, window=600.0),
    ))


def test_a_caller_flooding_query_is_limited():
    """Every refusal is held to the deadline, so without a limit one caller parks a
    task per request for the whole of it while the model queues behind them."""
    client = _limited(3)
    for _ in range(3):
        assert client.post("/query", json={"query": "hi"}, headers=as_user(1)).status_code == 200
    stopped = client.post("/query", json={"query": "hi"}, headers=as_user(1))
    assert stopped.status_code == 429
    assert "Retry-After" in stopped.headers


def test_an_answer_and_a_refusal_cost_the_same_allowance():
    """The §5 property. If a refusal were cheaper — or free — a caller could read
    their own remaining allowance as a signal about what they had just been told,
    which is the distinction the generic refusal and the padding both remove."""
    conflict = lambda s: {
        "hop_count": 1,
        "permission_conflicts": [PermConflict(
            document_id=2, source_platform="confluence", source_ref="COMPLIANCE/x",
            sensitivity="restricted", score_margin=0.2,
        )],
    }
    answers, refusals = _limited(3), _limited(3, retrieval=conflict)

    for client in (answers, refusals):
        for _ in range(3):
            assert client.post("/query", json={"query": "q"}, headers=as_user(1)).status_code == 200
        assert client.post("/query", json={"query": "q"}, headers=as_user(1)).status_code == 429

    # And the two paths really did differ in outcome, or the test proves nothing.
    fresh_a = _limited(9)
    fresh_r = _limited(9, retrieval=conflict)
    assert fresh_a.post("/query", json={"query": "q"}, headers=as_user(1)).json()["text"] != \
           fresh_r.post("/query", json={"query": "q"}, headers=as_user(1)).json()["text"]


def test_the_limit_is_per_caller_not_global():
    """One busy person must not lock everybody else out — that would turn a defence
    into the outage it exists to prevent."""
    from src.api.auth import SlidingWindow

    people = {1: ALEX, 2: UserContext(id=2, role="compliance", dept="compliance", clearance_level=1)}
    client = TestClient(create_app(
        nodes=_nodes(), user_loader=people.get,
        query_limit=SlidingWindow(limit=2, window=600.0),
    ))
    for _ in range(2):
        client.post("/query", json={"query": "q"}, headers=as_user(1))
    assert client.post("/query", json={"query": "q"}, headers=as_user(1)).status_code == 429
    assert client.post("/query", json={"query": "q"}, headers=as_user(2)).status_code == 200


def test_a_rejected_request_never_reads_the_question():
    """A 429 is decided from the caller's own request count and nothing else, so it
    cannot become a channel for anything about the corpus."""
    seen = []
    client = _limited(1, router=lambda s: seen.append(s["query"]) or {"route": "rag"})
    client.post("/query", json={"query": "first"}, headers=as_user(1))
    client.post("/query", json={"query": "restricted secret"}, headers=as_user(1))
    assert seen == ["first"]
