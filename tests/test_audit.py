"""Escalation, the audit hash chain, and the compliance route guards. No database."""

from datetime import UTC, datetime, timedelta
from functools import partial

import pytest
from fastapi.testclient import TestClient

from src.agents.audit import events_for, row_hash, verify_rows
from src.agents.escalation import escalation_node, escalation_reason
from src.agents.knowledge_gap import cluster
from src.agents.router import route_node
from src.agents.synthesizer import synthesize_node
from src.agents.verifier import verify_node
from src.api.main import create_app
from src.config import get_settings
from src.graph.state import (
    GENERIC_REFUSAL,
    Chunk,
    Citation,
    PermConflict,
    UserContext,
    VerificationResult,
)
from src.llm.fake import FakeChatModel

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)
MARCUS = UserContext(id=2, role="compliance", dept="compliance", clearance_level=1)
CONFLICT = PermConflict(
    document_id=7,
    source_platform="confluence",
    source_ref="COMPLIANCE/aml-escalation",
    sensitivity="restricted",
    score_margin=0.31,
)


@pytest.fixture(autouse=True)
def pinned_settings(monkeypatch):
    monkeypatch.setenv("VERIFIER_CONFIDENCE_THRESHOLD", "0.6")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _state(**over):
    return {
        "request_id": "00000000-0000-0000-0000-000000000001",
        "query": "q",
        "user": ALEX,
        "route": "rag",
        "hop_count": 1,
        "retrieved_chunks": [],
        "permission_conflicts": [],
        "sql_result": None,
        "draft_answer": None,
        "verification": None,
        "clarification_question": None,
        "final_answer": None,
        "citations": [],
        "explanation": None,
        "escalated": False,
        "audit_events": [],
        **over,
    }


# --- Escalation ---------------------------------------------------------------

@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"permission_conflicts": [CONFLICT]}, "permission_conflict"),
        ({}, "insufficient_evidence"),
        ({"draft_answer": "x", "verification": VerificationResult(grounded=True, confidence=0.3)},
         "low_confidence"),
        ({"draft_answer": "x", "hop_count": 3,
          "verification": VerificationResult(grounded=False, unsupported=["c"], confidence=0.9)},
         "unsupported"),
    ],
)
def test_every_cause_refuses_with_the_same_text_and_its_own_reason(over, reason):
    assert escalation_reason(_state(**over)) == reason
    out = escalation_node(_state(**over))
    assert out["escalated"] is True
    assert out["final_answer"] == GENERIC_REFUSAL
    assert out["citations"] == []
    assert out["audit_events"][-1].payload["reason"] == reason


def test_the_explanation_names_the_withheld_item_for_the_auditor():
    out = escalation_node(_state(permission_conflicts=[CONFLICT]))
    assert "COMPLIANCE/aml-escalation" in out["explanation"]
    assert "clearance 0" in out["explanation"]


def test_the_explanation_never_reaches_the_asker_through_the_real_node():
    """Real Escalation node in the real graph; only the chat model and retrieval stood in."""
    chat = FakeChatModel(route="rag")
    nodes = {
        "router": partial(route_node, chat_model=chat),
        "clarification": lambda s: {},
        "synthesizer": partial(synthesize_node, chat_model=chat),
        "verifier": partial(verify_node, chat_model=chat),
        "sql_tool": lambda s: {},
        "retrieval": lambda s: {"hop_count": 1, "permission_conflicts": [CONFLICT]},
        "escalation": escalation_node,
        "audit": lambda s: {"audit_events": []},
    }
    r = TestClient(create_app(nodes=nodes, user_loader=lambda uid: ALEX)).post(
        "/query", json={"query": "What triggers an AML escalation review?", "user_id": 1}
    )
    assert r.json() == {"text": GENERIC_REFUSAL, "citations": []}


# --- The trail ----------------------------------------------------------------

def test_the_trail_records_identity_but_never_chunk_content():
    chunk = Chunk(
        id=10, document_id=1, content="SECRET BODY TEXT", acl_tags=["support"], score=0.8,
        citation=Citation(document_id=1, title="t", source_platform="confluence", source_ref="r"),
    )
    trail = events_for(_state(retrieved_chunks=[chunk], permission_conflicts=[CONFLICT]))
    types = [e.event_type for e in trail]
    assert types[0] == "query_received" and types[-1] == "final_answer"
    assert "permission_conflict" in types
    assert "SECRET BODY TEXT" not in str([e.payload for e in trail])


def test_escalation_reason_and_explanation_are_carried_into_the_trail():
    escalated = _state(permission_conflicts=[CONFLICT])
    escalated.update(escalation_node(escalated))
    trail = events_for(escalated)
    escalation = next(e for e in trail if e.event_type == "escalation")
    assert escalation.payload["reason"] == "permission_conflict"
    assert "COMPLIANCE/aml-escalation" in escalation.payload["explanation"]


# --- Hash chain ---------------------------------------------------------------

def _chain(n=4):
    rows, prev, t0 = [], None, datetime(2026, 10, 1, tzinfo=UTC)
    for i in range(n):
        args = ("00000000-0000-0000-0000-000000000001", "retrieval", 1, {"i": i, "f": 0.31},
                t0 + timedelta(seconds=i))
        digest = row_hash(prev, *args)
        rows.append([i + 1, args[0], args[1], args[2], args[3], prev, digest, args[4]])
        prev = digest
    return rows


def test_an_untouched_chain_verifies():
    report = verify_rows(_chain())
    assert report.ok and report.rows_checked == 4


@pytest.mark.parametrize(
    ("column", "value"), [(4, {"i": 99, "f": 0.31}), (3, 2), (2, "final_answer")]
)
def test_editing_payload_user_or_type_is_flagged_at_that_row(column, value):
    rows = _chain()
    rows[2][column] = value
    report = verify_rows(rows)
    assert not report.ok and report.first_bad_id == 3


def test_deleting_a_row_is_flagged_at_the_next_one():
    rows = _chain()
    del rows[1]
    report = verify_rows(rows)
    assert not report.ok and report.first_bad_id == 3
    assert "removed" in report.problem


def test_the_hash_does_not_depend_on_the_session_timezone():
    t = datetime(2026, 10, 1, 12, tzinfo=UTC)
    sgt = t.astimezone(__import__("zoneinfo").ZoneInfo("Asia/Singapore"))
    assert row_hash(None, "r", "e", 1, {}, t) == row_hash(None, "r", "e", 1, {}, sgt)


# --- Knowledge gap clustering -------------------------------------------------

def test_similar_questions_cluster_and_distinct_ones_do_not():
    vectors = [[1, 0], [0.98, 0.2], [0, 1], [0.1, 0.99]]
    assert cluster(vectors, threshold=0.9) == [[0, 1], [2, 3]]


# --- Compliance route guards --------------------------------------------------

def _client(user):
    def unused_node(state):
        raise AssertionError("graph must not run")

    nodes = dict.fromkeys(
        ["router", "retrieval", "sql_tool", "clarification", "synthesizer", "verifier",
         "escalation", "audit"], unused_node,
    )
    return TestClient(create_app(nodes=nodes, user_loader=lambda uid: user))


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/audit/00000000-0000-0000-0000-000000000001", None),
        ("get", "/audit/verify", None),
        ("get", "/knowledge-gaps", None),
        ("post", "/admin/permissions/revoke",
         {"user_id": 1, "source_platform": "confluence", "source_ref": "x"}),
        ("post", "/admin/permissions/grant",
         {"user_id": 1, "source_platform": "confluence", "source_ref": "x"}),
    ],
)
@pytest.mark.parametrize("caller", [ALEX, None], ids=["not-an-officer", "unknown-user"])
def test_only_a_compliance_officer_reaches_compliance_routes(method, path, body, caller):
    client, headers = _client(caller), {"X-User-Id": "1"}
    r = client.post(path, headers=headers, json=body) if body else client.get(path, headers=headers)
    assert r.status_code == 403


def test_a_malformed_request_id_is_rejected_before_the_database():
    r = _client(MARCUS).get("/audit/not-a-uuid", headers={"X-User-Id": "2"})
    assert r.status_code == 422
