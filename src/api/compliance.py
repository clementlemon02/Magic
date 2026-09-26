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
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException
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


class ChainStatus(BaseModel):
    ok: bool
    rows_checked: int
    first_bad_id: int | None = None
    problem: str | None = None


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


def build_router(user_loader: Callable, connect: Callable = audit._connect) -> APIRouter:
    router = APIRouter()

    def officer(x_user_id: int = Header()) -> UserContext:
        user = user_loader(x_user_id)
        # Same answer for an unknown id and a non-officer: this is not a way to probe
        # which user ids exist.
        if user is None or user.role != COMPLIANCE_ROLE:
            raise HTTPException(status_code=403, detail="Compliance role required")
        return user

    def record(conn, user: UserContext, event_type: str, payload: dict) -> None:
        event = AuditEvent(event_type=event_type, payload=payload, occurred_at=datetime.now(UTC))
        audit.append_in(conn, str(uuid.uuid4()), user.id, [event])

    @router.get("/audit/verify", response_model=ChainStatus)
    async def verify(user: UserContext = Depends(officer)) -> ChainStatus:
        report = await run_in_threadpool(audit.verify_audit_chain, connect)
        return ChainStatus(**report.__dict__)

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
