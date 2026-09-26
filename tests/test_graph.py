"""Topology tests for the request-time graph, driven with fake nodes.

These assert the routing decisions in CLAUDE.md §2.2 and §4 — no LLM, no database.
"""

from datetime import UTC, datetime

import pytest

from src.config import get_settings
from src.graph.graph import build_graph
from src.graph.state import (
    Chunk,
    GENERIC_REFUSAL,
    AuditEvent,
    Citation,
    PermConflict,
    UserContext,
    VerificationResult,
)


@pytest.fixture(autouse=True)
def pinned_settings(monkeypatch):
    """Pin the thresholds so a developer's local .env cannot change the outcome."""
    monkeypatch.setenv("RETRIEVAL_MAX_HOPS", "3")
    monkeypatch.setenv("VERIFIER_CONFIDENCE_THRESHOLD", "0.6")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _citation() -> Citation:
    return Citation(
        document_id=1,
        title="Customer Refund & Chargeback Policy",
        source_platform="confluence",
        source_ref="SUPPORT/refund-policy",
    )


def _chunk(i: int) -> Chunk:
    return Chunk(id=i, document_id=i, content=f"passage {i}", acl_tags=["support"], score=0.8,
                 citation=_citation())


def _conflict() -> PermConflict:
    return PermConflict(
        document_id=2,
        source_platform="confluence",
        source_ref="COMPLIANCE/aml-escalation",
        sensitivity="restricted",
        score_margin=0.2,
    )


def _audit(state) -> dict:
    return {
        "audit_events": [
            AuditEvent(event_type="final_answer", payload={}, occurred_at=datetime.now(UTC))
        ]
    }


def _start(**overrides) -> dict:
    state = {
        "request_id": "r-1",
        "query": "How long do customers have to contest a chargeback?",
        "user": UserContext(id=1, role="support", dept="support", clearance_level=0),
        "hop_count": 0,
        "retrieved_chunks": [],
        "permission_conflicts": [],
        "citations": [],
        "escalated": False,
        "audit_events": [],
        "clarification_question": None,
    }
    state.update(overrides)
    return state


def _graph(**nodes):
    base = {
        "router": lambda s: {"route": "rag"},
        "retrieval": lambda s: {"hop_count": s.get("hop_count", 0) + 1},
        "sql_tool": lambda s: {"sql_result": 42},
        "clarification": lambda s: {"clarification_question": "Which ticket?"},
        "synthesizer": lambda s: {"draft_answer": "45 days.", "citations": [_citation()]},
        "verifier": lambda s: {
            "verification": VerificationResult(grounded=True, unsupported=[], confidence=0.9)
        },
        "escalation": lambda s: {"escalated": True, "explanation": "restricted match"},
        "audit": _audit,
    }
    base.update(nodes)
    return build_graph(**base)


def test_rag_path_reaches_an_answer():
    out = _graph().invoke(_start())
    assert out["final_answer"] == "45 days."
    assert out["citations"][0].source_platform == "confluence"
    assert out["escalated"] is False


def test_permission_conflict_refuses_generically():
    """Scenario 3. The asker-facing text must name nothing restricted."""
    out = _graph(
        retrieval=lambda s: {
            "hop_count": s.get("hop_count", 0) + 1,
            "permission_conflicts": [_conflict()],
        }
    ).invoke(_start())
    assert out["escalated"] is True
    assert out["final_answer"] == GENERIC_REFUSAL
    assert out["citations"] == []
    # The specific reason survives for the compliance inquiry, but not in the answer.
    assert out["explanation"] == "restricted match"
    assert "aml" not in out["final_answer"].lower()


def test_ungrounded_answer_loops_then_stops_at_the_hop_cap():
    """The loop must terminate. Without the cap this runs until langgraph's recursion limit."""
    hops = []

    def counting_retrieval(s):
        hops.append(1)
        # New evidence every hop, so only the cap can stop the loop.
        return {"hop_count": s.get("hop_count", 0) + 1, "retrieved_chunks": [_chunk(len(hops))]}

    out = _graph(
        retrieval=counting_retrieval,
        verifier=lambda s: {
            "verification": VerificationResult(
                grounded=False, unsupported=["no source for 45 days"], confidence=0.8
            )
        },
    ).invoke(_start())

    assert len(hops) == 3, f"expected RETRIEVAL_MAX_HOPS=3 attempts, got {len(hops)}"
    assert out["escalated"] is True
    assert out["final_answer"] == GENERIC_REFUSAL


def test_a_retry_that_finds_the_same_evidence_stops_instead_of_redrafting():
    """Same chunks as the last hop means the same verdict; don't pay for it twice."""
    hops, drafts = [], []

    def same_every_time(s):
        hops.append(1)
        return {"hop_count": s.get("hop_count", 0) + 1, "retrieved_chunks": [_chunk(1)]}

    out = _graph(
        retrieval=same_every_time,
        synthesizer=lambda s: (drafts.append(1), {"draft_answer": "x"})[1],
        verifier=lambda s: {
            "verification": VerificationResult(grounded=False, unsupported=["x"], confidence=0.9)
        },
    ).invoke(_start())

    assert len(hops) == 2  # the repeat is noticed on the first retry
    assert len(drafts) == 1  # and never drafted or verified again
    assert out["escalated"] is True
    assert out["final_answer"] == GENERIC_REFUSAL


def test_low_confidence_escalates_even_when_grounded():
    """§4: confidence below the threshold goes to Escalation regardless of grounding."""
    out = _graph(
        verifier=lambda s: {
            "verification": VerificationResult(grounded=True, unsupported=[], confidence=0.4)
        }
    ).invoke(_start())
    assert out["escalated"] is True
    assert out["final_answer"] == GENERIC_REFUSAL


def test_sql_route_skips_retrieval_and_still_verifies():
    seen = []
    out = _graph(
        router=lambda s: {"route": "sql"},
        retrieval=lambda s: seen.append("retrieval") or {},
        synthesizer=lambda s: {"draft_answer": "128 transactions.", "citations": []},
    ).invoke(_start())
    assert seen == []
    assert out["sql_result"] == 42
    assert out["final_answer"] == "128 transactions."


def test_clarification_ends_the_turn_with_the_question():
    """§4's one round: the asker gets the question, and answers by asking again."""
    reached = []
    out = _graph(
        router=lambda s: {"route": "clarify"},
        retrieval=lambda s: reached.append("retrieval") or {},
        synthesizer=lambda s: reached.append("synthesizer") or {},
    ).invoke(_start())

    assert out["final_answer"] == "Which ticket?"
    assert out["citations"] == []
    assert out["escalated"] is False
    # No evidence was gathered and no LLM drafted anything for an unanswerable question.
    assert reached == []


def test_a_refusal_outranks_a_pending_clarification():
    """Escalation must win, so clarification cannot become a side channel (§5)."""
    out = _graph(
        router=lambda s: {"route": "clarify"},
        clarification=lambda s: {
            "clarification_question": "Which compliance space did you mean?",
            "escalated": True,
        },
    ).invoke(_start())
    assert out["final_answer"] == GENERIC_REFUSAL


def _ungrounded_loop(**start):
    hops = []

    def retrieval(s):
        hops.append(1)
        return {"hop_count": s.get("hop_count", 0) + 1, "retrieved_chunks": [_chunk(len(hops))]}

    out = _graph(
        retrieval=retrieval,
        verifier=lambda s: {
            "verification": VerificationResult(grounded=False, unsupported=["x"], confidence=0.9)
        },
    ).invoke(_start(**start))
    return hops, out


def test_no_new_hop_starts_once_the_time_budget_is_spent():
    """A refusal must end within budget + one hop, or the padding can't hide it."""
    import time

    hops, out = _ungrounded_loop(started_at=time.monotonic() - 60)
    assert len(hops) == 1
    assert out["escalated"] is True
    assert out["final_answer"] == GENERIC_REFUSAL


def test_hops_continue_while_budget_remains():
    import time

    hops, out = _ungrounded_loop(started_at=time.monotonic())
    assert len(hops) == 3  # the fake nodes are instant, so only the hop cap stops it
    assert out["escalated"] is True
