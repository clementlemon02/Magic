"""Single source of truth for graph state shape — CLAUDE.md §3.

Every agent reads and writes a subset of `GraphState`, nothing more. The response
contracts at the bottom (`AskerResponse`, `ComplianceExplanation`) are the shared
boundary between the orchestration and audit workstreams; the Technical Design's
risk table calls for locking them before either side starts.
"""

from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

SourcePlatform = Literal["confluence", "jira", "slack", "drive", "internal"]
Sensitivity = Literal["internal", "restricted"]
Route = Literal["rag", "sql", "clarify", "escalate"]
AuditEventType = Literal[
    "query_received",
    "retrieval",
    "sql_executed",
    "draft_answer",
    "verification",
    "permission_conflict",
    "escalation",
    "final_answer",
]

# The exact asker-facing refusal (CLAUDE.md §5). Lives here because Escalation
# writes it and the API layer asserts on it — a judged demo moment, so one string.
GENERIC_REFUSAL = "I don't have an answer you're permitted to see for this request."


class UserContext(BaseModel):
    """The caller. Crosses the API trust boundary, so it validates on construction."""

    id: int
    role: str
    dept: str
    clearance_level: int  # mirrors roles.clearance_level: 0 standard, 1 restricted

    def acl_tags(self) -> list[str]:
        """The tags this caller matches, for the §1 database-level filter.

        Defined once because retrieval and the SQL tool both build predicates from
        it — two copies of this would be two chances to disagree about who can see
        what. Callers pass the result as a query parameter; never interpolate it.
        """
        # "all-staff" because every caller is staff: without it, content a source
        # marks as company-wide (a public Slack channel, an all-staff Drive file)
        # matched nobody. A live grant in `permissions` is still required on top.
        return [self.role, self.dept, "all-staff"]


class Citation(BaseModel):
    """A pointer back to the source item a claim came from.

    `source_platform` + `source_ref` are what Scenario 1 resolves into each
    platform's native location, and they match `permissions.source_ref` so a
    citation can always be re-checked against the live ACL.
    """

    document_id: int
    title: str
    source_platform: SourcePlatform
    source_ref: str


class Chunk(BaseModel):
    """An ACL-filtered chunk. Only chunks the caller may see are ever built."""

    id: int
    document_id: int
    content: str
    acl_tags: list[str]
    score: float
    citation: Citation


class PermConflict(BaseModel):
    """A restricted item that outscored the best permitted result.

    Identity only, never content — this object is the reason Escalation knows to
    refuse, and it must be safe to hold in state that also feeds the answer path.
    Populated by the unfiltered arm of retrieval (CLAUDE.md §4).
    """

    model_config = ConfigDict(extra="forbid")

    document_id: int
    source_platform: SourcePlatform
    source_ref: str
    sensitivity: Sensitivity
    score_margin: float


class VerificationResult(BaseModel):
    grounded: bool
    unsupported: list[str] = Field(default_factory=list)
    confidence: float


class AuditEvent(BaseModel):
    """One node transition, destined for a hash-chained `audit_log` row (§6).

    `occurred_at` is set by the caller rather than defaulted in the database:
    `row_hash` is computed over it, so the writer has to know the value it hashed.
    """

    event_type: AuditEventType
    payload: dict[str, Any]
    occurred_at: datetime


class GraphState(TypedDict):
    request_id: str
    query: str
    user: UserContext
    route: Route
    hop_count: int
    retrieved_chunks: list[Chunk]      # ACL-filtered, used for the answer
    permission_conflicts: list[PermConflict]
    sql_result: Any | None
    draft_answer: str | None
    verification: VerificationResult | None
    clarification_question: str | None
    final_answer: str | None
    citations: list[Citation]
    explanation: str | None            # NEVER sent to the asker; compliance inquiry only
    escalated: bool
    audit_events: list[AuditEvent]


class AskerResponse(BaseModel):
    """Everything the asker is allowed to see.

    Deliberately has no field that can carry `explanation`, and forbids extras so
    one cannot be added at a call site. CLAUDE.md §5: widening this to explain a
    refusal is the exact failure the brief's negative case tests for.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[Citation] = Field(default_factory=list)


class ComplianceExplanation(BaseModel):
    """The compliance-officer view, reachable only via GET /audit/{request_id}."""

    request_id: str
    explanation: str | None
    permission_conflicts: list[PermConflict]
    verification: VerificationResult | None
    events: list[AuditEvent]
