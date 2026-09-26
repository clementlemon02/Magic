"""Query-time source recheck: the staleness window between a source revocation
and our next ingest (src/connectors/__init__.py).

The mirror — ACL tags plus a live `permissions` row — is only as fresh as the last
ingest. These cover the case that matters: the mirror still says yes about material
the source now protects.
"""

import pytest

from src.agents.retrieval import FILTERED_SEARCH, drop_source_revoked, retrieval_node
from src.connectors import connectors, source_denies
from src.connectors.confluence import ConfluenceConnector
from src.graph.state import Chunk, Citation, UserContext

AML = ("confluence", "COMPLIANCE/aml-escalation")   # restricted at the source
POLICY = ("confluence", "SUPPORT/refund-policy")    # internal at the source

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)
MARCUS = UserContext(id=2, role="compliance", dept="compliance", clearance_level=1)


def _chunk(platform: str, ref: str, chunk_id: int = 1) -> Chunk:
    return Chunk(
        id=chunk_id, document_id=chunk_id, content="...", acl_tags=["compliance"], score=0.9,
        citation=Citation(
            document_id=chunk_id, title="T", source_platform=platform, source_ref=ref
        ),
    )


class FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


@pytest.fixture
def source_revokes_marcus():
    """Revoke at the SOURCE only, leaving our mirror untouched — the stale window."""
    connector = connectors()["confluence"]
    permission = connector.permissions_for(
        next(i for i in connector.list_items() if i.source_ref == AML[1])
    )
    before = list(permission.viewer_groups)
    permission.viewer_groups.clear()
    yield
    permission.viewer_groups[:] = before


# --- the rules ---------------------------------------------------------------

def test_restricted_item_follows_the_live_source_acl():
    assert source_denies(ALEX, *AML) is True
    assert source_denies(MARCUS, *AML) is False


def test_internal_items_are_left_to_the_mirror():
    """The point of scoping this: no source round-trip on the common path."""
    assert source_denies(ALEX, *POLICY) is False


def test_a_platform_with_no_connector_is_left_to_the_mirror():
    """`internal` documents are authored here and have no upstream to ask."""
    assert source_denies(ALEX, "internal", "HOMEGROWN/1") is False


def test_an_item_the_source_no_longer_lists_is_denied():
    assert source_denies(MARCUS, "confluence", "DELETED/999") is True


def test_a_raising_connector_denies(monkeypatch):
    """A broken source degrades to no restricted answers, never to unjustified ones."""
    def boom(self, user, item):
        raise RuntimeError("source unreachable")

    monkeypatch.setattr(ConfluenceConnector, "check_access", boom)
    assert source_denies(MARCUS, *AML) is True


# --- the drop ----------------------------------------------------------------

def test_source_revocation_drops_the_chunk_the_mirror_still_allows(source_revokes_marcus):
    """THE case: our permissions table still grants it, the source no longer does."""
    kept, denied = drop_source_revoked(MARCUS, [_chunk(*AML)])
    assert kept == []
    assert denied == [{"source_platform": "confluence", "source_ref": AML[1]}]


def test_the_same_chunk_survives_while_the_source_still_grants_it():
    kept, denied = drop_source_revoked(MARCUS, [_chunk(*AML)])
    assert len(kept) == 1 and denied == []


def test_only_the_denied_chunk_is_dropped(source_revokes_marcus):
    chunks = [_chunk(*AML, chunk_id=1), _chunk(*POLICY, chunk_id=2)]
    kept, denied = drop_source_revoked(MARCUS, chunks)
    assert [c.id for c in kept] == [2]
    assert len(denied) == 1


def test_denied_records_identity_only_never_content(source_revokes_marcus):
    """A chunk withheld here must not be copied into the audit log."""
    _, denied = drop_source_revoked(MARCUS, [_chunk(*AML)])
    assert set(denied[0]) == {"source_platform", "source_ref"}


# --- the node ----------------------------------------------------------------

def _node_state(user: UserContext) -> dict:
    return {"query": "what triggers an AML review?", "user": user, "hop_count": 0}


def _execute_returning_aml(sql, params):
    if sql != FILTERED_SEARCH:
        return []
    return [{
        "id": 1, "document_id": 2, "content": "Tier 2 review at SGD 9,500.",
        "acl_tags": ["compliance"], "title": "AML Escalation Procedure",
        "source_platform": "confluence", "source_ref": AML[1], "score": 0.95,
    }]


def test_node_drops_the_chunk_and_records_why(source_revokes_marcus):
    out = retrieval_node(
        _node_state(MARCUS), embeddings=FakeEmbeddings(), execute=_execute_returning_aml
    )
    assert out["retrieved_chunks"] == []
    events = [e for e in out["audit_events"] if e.event_type == "source_recheck_denied"]
    assert len(events) == 1
    assert events[0].payload == {"items": [{"source_platform": "confluence", "source_ref": AML[1]}]}


def test_node_records_nothing_when_the_source_still_agrees():
    out = retrieval_node(
        _node_state(MARCUS), embeddings=FakeEmbeddings(), execute=_execute_returning_aml
    )
    assert len(out["retrieved_chunks"]) == 1
    assert [e for e in out["audit_events"] if e.event_type == "source_recheck_denied"] == []
