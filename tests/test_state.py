"""Guards on the shared contract in src/graph/state.py.

These cover the two properties other modules are allowed to assume: that the
asker-facing response cannot carry a reason, and that a permission conflict
cannot carry content.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.graph.state import (
    GENERIC_REFUSAL,
    AskerResponse,
    AuditEvent,
    Chunk,
    Citation,
    ComplianceExplanation,
    GraphState,
    PermConflict,
    UserContext,
    VerificationResult,
)


def _citation() -> Citation:
    return Citation(
        document_id=1,
        title="Customer Refund & Chargeback Policy",
        source_platform="confluence",
        source_ref="SUPPORT/refund-policy",
    )


def _conflict() -> PermConflict:
    return PermConflict(
        document_id=2,
        source_platform="confluence",
        source_ref="COMPLIANCE/aml-escalation",
        sensitivity="restricted",
        score_margin=0.11,
    )


def test_asker_response_cannot_carry_an_explanation():
    """CLAUDE.md §5 — the refusal must not leak why, even by accident."""
    with pytest.raises(ValidationError):
        AskerResponse(
            text=GENERIC_REFUSAL,
            explanation="matched a Compliance-only AML escalation procedure",
        )


def test_generic_refusal_names_nothing_restricted():
    """Scenario 3 is judged on this string mentioning no topic, source or document."""
    lowered = GENERIC_REFUSAL.lower()
    for leak in ("aml", "compliance", "restricted", "confluence", "permission"):
        assert leak not in lowered


def test_perm_conflict_carries_identity_not_content():
    """Retrieval populates these from the UNFILTERED search — content must not ride along."""
    assert "content" not in PermConflict.model_fields
    with pytest.raises(ValidationError):
        PermConflict(
            document_id=2,
            source_platform="confluence",
            source_ref="COMPLIANCE/aml-escalation",
            sensitivity="restricted",
            score_margin=0.11,
            content="Transactions above SGD 10,000 are escalated to...",
        )


def test_full_state_round_trips():
    """A populated GraphState builds — catches drift between the nested models."""
    state: GraphState = {
        "request_id": "3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90",
        "query": "What is our chargeback window?",
        "user": UserContext(id=1, role="support", dept="support", clearance_level=0),
        "route": "rag",
        "hop_count": 1,
        "retrieved_chunks": [
            Chunk(
                id=10,
                document_id=1,
                content="Chargebacks must be contested within 45 days.",
                acl_tags=["support", "all-staff"],
                score=0.88,
                citation=_citation(),
            )
        ],
        "permission_conflicts": [_conflict()],
        "sql_result": None,
        "draft_answer": "45 days.",
        "verification": VerificationResult(grounded=True, unsupported=[], confidence=0.91),
        "clarification_question": None,
        "final_answer": "45 days.",
        "citations": [_citation()],
        "explanation": None,
        "escalated": False,
        "audit_events": [
            AuditEvent(
                event_type="query_received",
                payload={"query": "What is our chargeback window?"},
                occurred_at=datetime.now(UTC),
            )
        ],
    }
    assert state["retrieved_chunks"][0].citation.source_platform == "confluence"

    # The compliance view is the only place explanation and conflicts surface together.
    view = ComplianceExplanation(
        request_id=state["request_id"],
        explanation="Top match was a Compliance-only page; caller clearance 0.",
        permission_conflicts=state["permission_conflicts"],
        verification=state["verification"],
        events=state["audit_events"],
    )
    assert view.permission_conflicts[0].sensitivity == "restricted"
