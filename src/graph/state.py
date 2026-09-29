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
Route = Literal["rag", "sql", "clarify", "decline", "escalate"]
AuditEventType = Literal[
    "query_received",
    # One per node the graph ran, written as it ran (§4). The rows below are a
    # summary of the finished request; these are the request happening.
    "node_transition",
    "retrieval",
    "sql_executed",
    "draft_answer",
    "verification",
    "permission_conflict",
    "source_recheck_denied",
    "escalation",
    "final_answer",
    # Outside the graph: a request served from the answer cache, and the
    # compliance/admin actions, which are as much a part of the trail as a query.
    "cache_hit",
    "compliance_inquiry",
    "permission_revoked",
    "permission_granted",
]
# Why a request was refused. Stored in `escalations.reason` and in the audit trail,
# never shown to the asker (§5).
EscalationReason = Literal[
    "permission_conflict",    # a restricted item outscored everything permitted
    "insufficient_evidence",  # nothing permitted to draft from
    "unsupported",            # draft still ungrounded at the hop cap
    "low_confidence",         # Verifier below VERIFIER_CONFIDENCE_THRESHOLD
]

# The exact asker-facing refusal (CLAUDE.md §5). Lives here because Escalation
# writes it and the API layer asserts on it — a judged demo moment, so one string.
GENERIC_REFUSAL = "I don't have an answer you're permitted to see for this request."

# The `decline` route's reply: the message was not a question about this company's
# knowledge at all. Distinct from GENERIC_REFUSAL on purpose, and safe to be
# distinct: the Router picks this from the query TEXT alone, before retrieval and
# before any permission check, so it carries nothing about the corpus or the
# caller's access. GENERIC_REFUSAL is the one that must stay uniform, because that
# one IS decided by what the caller may see.
DECLINE_REPLY = (
    "I answer questions about Aurelia's internal knowledge — documents in Confluence, "
    "Jira, Slack and Drive, and figures from the transactions table. Ask me something "
    "from there and I'll answer what your permissions allow."
)


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

    document_id: int | None  # None for a transactions query, which is not a document
    title: str
    source_platform: SourcePlatform
    source_ref: str

    # What actually backs the claim, so an asker can check the answer instead of
    # trusting it: the passage the wording came from, or the query that counted it.
    #
    # Safe here and NOT on AskerResponse, which §5 keeps unable to carry anything:
    # `build_response` gives a refusal `AskerResponse(text=GENERIC_REFUSAL)` and
    # nothing else, so `citations` is empty on every refusal and there is no path
    # for either field to ride out on one. Both hold material the caller has already
    # been shown in summary — every chunk reaching here passed the §1 filter.
    passage: str | None = None
    query: str | None = None


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
    started_at: float                  # time.monotonic() when the graph began; bounds retries
    retrieved_chunks: list[Chunk]      # ACL-filtered, used for the answer
    evidence_exhausted: bool           # this hop retrieved the same chunks as the last
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


class Stage(BaseModel):
    """One node the graph ran, how long it took, and what it did.

    `detail` is a few words for the pipeline view — "8 passages", "grounded 0.94".
    Safe because a Stage only ever reaches an asker on an ANSWER: `build_response`
    gives a refusal an empty trace, so nothing here can describe a withheld request.
    Counts permitted evidence only, never how much was filtered out — that number
    would say restricted material exists, which is the §5 leak in another costume.
    """

    node: str
    hop: int
    ms: float
    detail: str | None = None


class AskerResponse(BaseModel):
    """Everything the asker is allowed to see.

    Deliberately has no field that can carry `explanation`, and forbids extras so
    one cannot be added at a call site. CLAUDE.md §5: widening this to explain a
    refusal is the exact failure the brief's negative case tests for.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[Citation] = Field(default_factory=list)

    # What the graph did, for the asker to watch it work. Answers only, and for the
    # same reason citations are: `build_response` gives a refusal GENERIC_REFUSAL and
    # nothing else, so this is empty on every refusal by construction.
    #
    # It must stay that way. A trace on a refusal reconstructs the cause the wording
    # withholds — a permission conflict short-circuits at retrieval on hop 1, an
    # unsupported answer grinds through three — which is the same leak that made the
    # refusal constant-time in the first place. Streaming these live would leak it
    # too, over the network rather than in the payload; the UI replays them after the
    # response instead, so the clock outside is unchanged.
    trace: list[Stage] = Field(default_factory=list)


class ComplianceExplanation(BaseModel):
    """The compliance-officer view, reachable only via GET /audit/{request_id}."""

    request_id: str
    explanation: str | None
    permission_conflicts: list[PermConflict]
    verification: VerificationResult | None
    events: list[AuditEvent]
