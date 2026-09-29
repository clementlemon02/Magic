"""Topology tests for the request-time graph, driven with fake nodes.

These assert the routing decisions in CLAUDE.md §2.2 and §4 — no LLM, no database.
"""

from datetime import UTC, datetime

import pytest

from src.config import get_settings
from src.graph.graph import build_graph
from src.graph.state import (
    DECLINE_REPLY,
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


# --- one row per node transition (§4) ------------------------------------------

def _capturing_audit():
    """An audit node that keeps what it was handed.

    The fixture's `_audit` returns a fresh list, which is what the real one does too
    (it returns the assembled trail) — so reading the FINAL state shows only that.
    What matters is what reached the audit node, since that is what it writes.
    """
    seen: dict[str, list] = {"events": []}

    def audit(state) -> dict:
        seen["events"] = list(state.get("audit_events") or [])
        return {"audit_events": seen["events"]}

    return audit, seen


def _transitions(events) -> list[dict]:
    return [e.payload for e in events if e.event_type == "node_transition"]


def test_every_node_the_graph_ran_leaves_a_row():
    audit, seen = _capturing_audit()
    _graph(audit=audit).invoke(_start())
    ran = [t["node"] for t in _transitions(seen["events"])]
    assert ran == ["router", "retrieval", "synthesizer", "verifier", "answer"]
    # `audit` writes the trail, so it can never be a row inside it.
    assert "audit" not in ran


def test_each_hop_of_a_multi_hop_request_is_on_the_record():
    """The defect this replaced: three hops recorded the last one and a count."""
    audit, seen = _capturing_audit()
    _graph(
        audit=audit,
        verifier=lambda s: {
            "verification": VerificationResult(
                grounded=False, unsupported=["unsupported claim"], confidence=0.9
            )
        },
        retrieval=lambda s: {
            "hop_count": s.get("hop_count", 0) + 1,
            "retrieved_chunks": [_chunk(s.get("hop_count", 0))],
        },
    ).invoke(_start())
    hops = [t["hop"] for t in _transitions(seen["events"]) if t["node"] == "retrieval"]
    assert hops == [1, 2, 3], "every hop should have its own row, not just the last"


def test_a_transition_row_carries_its_own_duration():
    audit, seen = _capturing_audit()
    _graph(audit=audit).invoke(_start())
    rows = _transitions(seen["events"])
    assert rows and all(isinstance(t["ms"], float) and t["ms"] >= 0 for t in rows)


def test_a_nodes_own_events_survive_the_recording_wrapper():
    """Retrieval and Escalation append their own events; the wrapper must add to
    whatever came back, not replace it."""
    marker = AuditEvent(
        event_type="source_recheck_denied", payload={"items": []}, occurred_at=datetime.now(UTC)
    )
    audit, seen = _capturing_audit()
    _graph(
        audit=audit,
        retrieval=lambda s: {
            "hop_count": s.get("hop_count", 0) + 1,
            "audit_events": [*(s.get("audit_events") or []), marker],
        },
    ).invoke(_start())
    assert marker in seen["events"]
    assert _transitions(seen["events"]), "the wrapper's own row is still there too"


def test_decline_ends_the_turn_without_touching_the_corpus():
    """Not a question about company knowledge. Decided by the Router from the query
    text alone, so nothing is retrieved and nothing is drafted."""
    reached = []
    out = _graph(
        router=lambda s: {"route": "decline"},
        retrieval=lambda s: reached.append("retrieval") or {},
        synthesizer=lambda s: reached.append("synthesizer") or {},
        clarification=lambda s: reached.append("clarification") or {},
    ).invoke(_start())

    assert out["final_answer"] == DECLINE_REPLY
    assert out["citations"] == []
    assert out["escalated"] is False
    assert reached == []


def test_a_refusal_outranks_a_decline():
    """Same ordering rule as the clarification case, for the same reason. A `decline`
    is a distinguishable reply, so if it could be reached after the permission filter
    ran it would be a side channel; checking it below the escalation branch is what
    makes 'it never can' true rather than merely likely."""
    out = _graph(router=lambda s: {"route": "decline"},
                 escalation=lambda s: {"escalated": True}).invoke(
        _start(route="decline", escalated=True)
    )
    assert out["final_answer"] == GENERIC_REFUSAL


def test_the_two_fixed_replies_stay_distinct_and_neither_contains_the_other():
    """DECLINE_REPLY may differ from GENERIC_REFUSAL — the Router picks it before any
    permission check, so it carries nothing about the corpus. What must not happen is
    the two drifting together: a refusal that reads like a scope message, or a scope
    message that hints at withheld material."""
    assert DECLINE_REPLY != GENERIC_REFUSAL
    assert GENERIC_REFUSAL not in DECLINE_REPLY and DECLINE_REPLY not in GENERIC_REFUSAL
    for tell in ("permitted", "restricted", "clearance", "access", "not allowed"):
        assert tell not in DECLINE_REPLY.lower(), f"the scope message hints at withholding: {tell}"


def test_a_stage_detail_counts_only_permitted_evidence():
    """"6 passages" is what this caller got. How many were filtered OUT is the number
    §5 forbids — it would tell an asker restricted material exists without naming it,
    which is the existence leak in another costume. `_detail` reads the node's own
    output, which only ever holds what survived the ACL predicate."""
    from src.graph.graph import _detail

    assert _detail("retrieval", {"retrieved_chunks": [_chunk(1), _chunk(2)]}) == "2 passages"
    assert _detail("retrieval", {"retrieved_chunks": [_chunk(1)]}) == "1 passage"
    assert _detail("retrieval", {"retrieved_chunks": []}) is None
    assert _detail("verifier", {"verification": VerificationResult(
        grounded=True, unsupported=[], confidence=0.94)}) == "grounded 0.94"
    assert _detail("escalation", {"escalated": True, "explanation": "matched COMPLIANCE/aml"}) is None


def test_a_detail_never_rides_out_on_a_refusal():
    """The whole reason per-node detail is safe: build_response gives a refusal an
    empty trace, so nothing computed here can describe a withheld request."""
    from src.graph.graph import build_response

    ran = [AuditEvent(event_type="node_transition",
                      payload={"node": "retrieval", "hop": 1, "ms": 5.0, "detail": "6 passages"},
                      occurred_at=datetime.now(UTC))]
    assert build_response({"escalated": True, "audit_events": ran}).trace == []
    kept = build_response({"escalated": False, "final_answer": "x", "audit_events": ran})
    assert kept.trace[0].detail == "6 passages"
