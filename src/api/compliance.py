"""Compliance inquiry and revocation admin — CLAUDE.md §4, §5, §7.

The only routes that can read `explanation` back out, so the only ones gated on the
caller's role. Every call here is itself written to the audit chain: reading why
someone was refused, or changing what they may see, is as auditable as the query.

ponytail: the caller is identified by an `X-User-Id` header resolved against the
database, the same trust level as /query's `user_id`. There is no authentication in
the MVP; put SSO in front of both before this leaves a demo.
"""

import uuid
from collections.abc import Callable
from functools import partial
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.agents import audit
from src.agents.access_gap import AccessGapReport, run_access_gap_scan
from src.agents.knowledge_gap import GapReport, run_knowledge_gap_scan
from src.graph.state import (
    AuditEvent,
    ComplianceExplanation,
    PermConflict,
    SourcePlatform,
    UserContext,
    VerificationResult,
)

COMPLIANCE_ROLE = "compliance"


class PermissionChange(BaseModel):
    model_config = {"extra": "forbid"}

    user_id: int
    source_platform: SourcePlatform
    source_ref: str


class PermissionChangeResult(BaseModel):
    changed: int


class RequestSummary(BaseModel):
    """One request in the officer's worklist. Carries no explanation — that is still
    only readable one request at a time, through /audit/{request_id}, and that read is
    itself recorded."""

    request_id: str
    at: datetime
    user_id: int | None
    query: str
    route: str | None
    escalated: bool
    reason: str | None


class SourceStatus(BaseModel):
    """One connector, as the admin screen shows it."""

    platform: SourcePlatform
    documents: int
    chunks: int
    restricted: int
    grants: int
    last_ingested: datetime | None


class Grant(BaseModel):
    source_platform: SourcePlatform
    source_ref: str
    sensitivity: str | None
    granted_at: datetime
    # None while live. Set when revoked, so the admin screen can offer it back.
    revoked_at: datetime | None = None


class ChainStatus(BaseModel):
    ok: bool
    rows_checked: int
    first_bad_id: int | None = None
    problem: str | None = None


SOURCES_SQL = """
    SELECT d.source_platform,
           count(DISTINCT d.id)                                             AS documents,
           count(c.id)                                                      AS chunks,
           count(DISTINCT d.id) FILTER (WHERE d.sensitivity = 'restricted') AS restricted,
           max(d.created_at)                                                AS last_ingested
    FROM documents AS d
    LEFT JOIN document_chunks AS c ON c.document_id = d.id
    GROUP BY d.source_platform
    ORDER BY d.source_platform
"""

GRANT_COUNTS_SQL = """
    SELECT source_platform, count(*) AS grants
    FROM permissions WHERE revoked_at IS NULL
    GROUP BY source_platform
"""

# A person's live grants. Sensitivity comes from the document, so the screen can warn
# before someone revokes the one thing an officer actually needs.
# Revoked rows are listed TOO, so a revoke can be undone from the screen that made
# it. Before this the row simply vanished and the page's own grant path was
# unreachable — an admin tool whose primary destructive action had no way back.
#
# LISTING ONLY. Nothing authorizes off this query: retrieval and the cache
# fingerprint each run their own `revoked_at IS NULL` predicate (CLAUDE.md §1), and
# this must never become the thing that decides what someone may read.
GRANTS_FOR_SQL = """
    SELECT p.source_platform, p.source_ref, d.sensitivity, p.granted_at, p.revoked_at
    FROM permissions AS p
    LEFT JOIN documents AS d
      ON d.source_platform = p.source_platform AND d.source_ref = p.source_ref
    WHERE p.user_id = %(user_id)s
    ORDER BY (p.revoked_at IS NOT NULL), d.sensitivity DESC NULLS LAST,
             p.source_platform, p.source_ref
"""

REVOKE_SQL = """
    UPDATE permissions SET revoked_at = now()
    WHERE user_id = %(user_id)s AND source_platform = %(source_platform)s
      AND source_ref = %(source_ref)s AND revoked_at IS NULL
"""

# A no-op when a live grant already exists, so granting twice doesn't stack rows.
GRANT_SQL = """
    INSERT INTO permissions (user_id, source_platform, source_ref)
    SELECT %(user_id)s, %(source_platform)s, %(source_ref)s
    WHERE NOT EXISTS (
        SELECT 1 FROM permissions
        WHERE user_id = %(user_id)s AND source_platform = %(source_platform)s
          AND source_ref = %(source_ref)s AND revoked_at IS NULL
    )
"""


def _explanation_from(request_id: str, events: list[AuditEvent]) -> ComplianceExplanation:
    by_type = {e.event_type: e.payload for e in events}
    conflicts = by_type.get("permission_conflict", {}).get("conflicts", [])
    verification = by_type.get("verification")
    return ComplianceExplanation(
        request_id=request_id,
        explanation=by_type.get("escalation", {}).get("explanation"),
        permission_conflicts=[PermConflict(**c) for c in conflicts],
        verification=VerificationResult(**verification) if verification else None,
        events=events,
    )


def build_router(
    user_loader: Callable, connect: Callable = audit._connect, *, officer: Callable | None = None
) -> APIRouter:
    """`officer` comes from src/api/auth.py, so one place decides who an officer is.

    The fallback below exists only for a caller that builds this router on its own;
    create_app always passes the real one.
    """
    router = APIRouter()

    if officer is None:  # pragma: no cover - create_app always supplies it
        from src.api.auth import build_dependencies

        _, officer = build_dependencies(user_loader)

    def record(conn, user: UserContext, event_type: str, payload: dict) -> None:
        event = AuditEvent(event_type=event_type, payload=payload, occurred_at=datetime.now(UTC))
        audit.append_in(conn, str(uuid.uuid4()), user.id, [event])

    @router.get("/audit/verify", response_model=ChainStatus)
    async def verify(user: UserContext = Depends(officer)) -> ChainStatus:
        report = await run_in_threadpool(audit.verify_audit_chain, connect)
        return ChainStatus(**report.__dict__)

    @router.get("/audit/recent", response_model=list[RequestSummary])
    async def recent(
        days: int = 7,
        limit: int = 100,
        outcome: str = "all",
        q: str = Query(default="", max_length=200),
        user: UserContext = Depends(officer),
    ):
        # Declared before /audit/{request_id}: FastAPI matches in order, and "recent"
        # would otherwise be taken for a request id and 422 on the UUID parse.
        #
        # `outcome` and `q` narrow in SQL, before the limit — see recent_requests. A
        # page that filtered its own fetched window would report "12 refusals" when
        # the window held 12 of 122.
        since = datetime.now(UTC) - timedelta(days=days)
        rows = await run_in_threadpool(
            partial(audit.recent_requests, since, limit=min(limit, 500),
                    outcome=outcome, q=q, connect=connect)
        )
        return [RequestSummary(**row) for row in rows]

    @router.get("/audit/{request_id}", response_model=ComplianceExplanation)
    async def inquiry(request_id: uuid.UUID, user: UserContext = Depends(officer)):
        def read_and_record() -> list[AuditEvent]:
            events = audit.read_request(str(request_id), connect)
            if events:
                # Reading why someone was refused is itself on the record.
                with connect() as conn, conn.transaction():
                    record(conn, user, "compliance_inquiry", {"request_id": str(request_id)})
            return events

        events = await run_in_threadpool(read_and_record)
        if not events:
            raise HTTPException(status_code=404, detail="Unknown request")
        return _explanation_from(str(request_id), events)

    def change(sql: str, event_type: str, body: PermissionChange, user: UserContext) -> int:
        with connect() as conn, conn.transaction():
            changed = conn.execute(sql, body.model_dump()).rowcount
            record(conn, user, event_type, {**body.model_dump(), "changed": changed})
        return changed

    @router.get("/admin/sources", response_model=list[SourceStatus])
    async def sources(user: UserContext = Depends(officer)):
        def read() -> list[SourceStatus]:
            with connect() as conn:
                rows = conn.execute(SOURCES_SQL).fetchall()
                counts = dict(conn.execute(GRANT_COUNTS_SQL).fetchall())
            return [
                SourceStatus(
                    platform=platform, documents=documents, chunks=chunks,
                    restricted=restricted, grants=counts.get(platform, 0),
                    last_ingested=last_ingested,
                )
                for platform, documents, chunks, restricted, last_ingested in rows
            ]

        return await run_in_threadpool(read)

    @router.get("/admin/permissions", response_model=list[Grant])
    async def grants(user_id: int, user: UserContext = Depends(officer)):
        def read() -> list[Grant]:
            with connect() as conn:
                rows = conn.execute(GRANTS_FOR_SQL, {"user_id": user_id}).fetchall()
            return [
                Grant(source_platform=p, source_ref=r, sensitivity=s,
                      granted_at=g, revoked_at=v)
                for p, r, s, g, v in rows
            ]

        return await run_in_threadpool(read)

    @router.post("/admin/permissions/revoke", response_model=PermissionChangeResult)
    async def revoke(body: PermissionChange, user: UserContext = Depends(officer)):
        # Takes effect on the next query with no other step: retrieval joins the live
        # permissions table (§1) and the answer cache is keyed on the grant set.
        changed = await run_in_threadpool(change, REVOKE_SQL, "permission_revoked", body, user)
        return PermissionChangeResult(changed=changed)

    @router.post("/admin/permissions/grant", response_model=PermissionChangeResult)
    async def grant(body: PermissionChange, user: UserContext = Depends(officer)):
        changed = await run_in_threadpool(change, GRANT_SQL, "permission_granted", body, user)
        return PermissionChangeResult(changed=changed)

    @router.get("/knowledge-gaps", response_model=GapReport)
    async def knowledge_gaps(days: int = 7, user: UserContext = Depends(officer)):
        since = datetime.now(UTC) - timedelta(days=days)
        return await run_in_threadpool(run_knowledge_gap_scan, since, connect=connect)

    @router.get("/access-gaps", response_model=AccessGapReport)
    async def access_gaps(days: int = 7, user: UserContext = Depends(officer)):
        # Officer-gated like the rest of this router, and for a stronger reason: the
        # report names restricted items and who asked for them. See access_gap.py.
        since = datetime.now(UTC) - timedelta(days=days)
        return await run_in_threadpool(run_access_gap_scan, since, connect=connect)

    return router
