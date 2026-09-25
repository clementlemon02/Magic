"""Permission-Conflict + Escalation — CLAUDE.md §4, §5.

Every path that ends without a grounded answer comes through here. It writes two
things that must never be confused:

- `final_answer`: GENERIC_REFUSAL, byte-for-byte the same whatever the cause.
- `explanation`: the specific reason, for the audit trail only. GET /audit/{id} is
  the one route that reads it back out.

No LLM call: the reason is decided from state, so it cannot be talked out of, and
the refusal path adds no model latency for the timing padding to absorb.

The `escalations` row is written by the audit node, in the same transaction as the
request's audit rows, so a refusal is never recorded in one table and not the other.
"""

from datetime import UTC, datetime

from src.config import get_settings
from src.graph.state import GENERIC_REFUSAL, AuditEvent, EscalationReason, GraphState


def escalation_reason(state: GraphState) -> EscalationReason:
    """Why this request cannot be answered. Checked in the order the graph can fail."""
    if state.get("permission_conflicts"):
        return "permission_conflict"
    verification = state.get("verification")
    if verification is None or not state.get("draft_answer"):
        return "insufficient_evidence"
    if verification.confidence < get_settings().verifier_confidence_threshold:
        return "low_confidence"
    return "unsupported"


def _explain(state: GraphState, reason: EscalationReason) -> str:
    user = state["user"]
    caller = (
        f"caller user {user.id} ({user.role}/{user.dept}, clearance {user.clearance_level})"
    )
    hops = state.get("hop_count", 0)

    if reason == "permission_conflict":
        items = "; ".join(
            f"{c.source_platform}:{c.source_ref} [{c.sensitivity}, +{c.score_margin:.2f}]"
            for c in state["permission_conflicts"]
        )
        return (
            f"Withheld: {len(state['permission_conflicts'])} restricted item(s) outranked "
            f"all permitted evidence — {items}. The {caller} holds no live grant for them."
        )
    if reason == "insufficient_evidence":
        return f"No permitted evidence supported an answer after {hops} hop(s), for the {caller}."
    verification = state["verification"]
    if reason == "low_confidence":
        return (
            f"Verifier confidence {verification.confidence:.2f} is below the "
            f"{get_settings().verifier_confidence_threshold:.2f} threshold, for the {caller}."
        )
    claims = "; ".join(verification.unsupported) or "unspecified"
    return f"Draft still unsupported after {hops} hop(s). Unsupported claims: {claims}."


def escalation_node(state: GraphState) -> dict:
    reason = escalation_reason(state)
    explanation = _explain(state, reason)
    event = AuditEvent(
        event_type="escalation",
        payload={"reason": reason, "explanation": explanation},
        occurred_at=datetime.now(UTC),
    )
    return {
        "escalated": True,
        "final_answer": GENERIC_REFUSAL,
        "citations": [],
        "explanation": explanation,
        "audit_events": [*(state.get("audit_events") or []), event],
    }
