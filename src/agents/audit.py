"""Audit — the tamper-evident trail, CLAUDE.md §4, §6.

Every request leaves rows in `audit_log`, each hash-chained to the one before it:

    row_hash = SHA-256(canonical JSON of [prev_hash, request_id, event_type,
                                          user_id, payload, created_at])

`user_id` is hashed as well as the columns §6 lists: otherwise a row could be
re-attributed to another user without breaking the chain.

Single writer, not a blockchain: a transaction-scoped advisory lock serialises
appends, so two requests can never both chain onto the same predecessor.

    python -m src.agents.audit verify    # exit 1 and the first bad row if tampered

ponytail: this detects an edited, inserted or deleted row anywhere in the chain,
but not the newest rows being truncated off the end — nothing in the table can.
Anchor the latest row_hash somewhere external (a signed daily digest) if that
matters.
"""

import hashlib
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.graph.state import AuditEvent, GraphState

# Held for the length of one append transaction. Arbitrary but fixed.
_CHAIN_LOCK = 0x41554454  # "AUDT"


def _canonical(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload exactly as JSONB will hand it back, so the hash recomputes."""
    return json.loads(json.dumps(payload, default=str))


def _utc(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat()


def row_hash(
    prev_hash: str | None,
    request_id: str,
    event_type: str,
    user_id: int | None,
    payload: dict[str, Any],
    created_at: datetime,
) -> str:
    material = json.dumps(
        [prev_hash, str(request_id), event_type, user_id, payload, _utc(created_at)],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _connect():
    import psycopg

    from src.config import get_settings

    settings = get_settings()
    return psycopg.connect(
        settings.database_url, connect_timeout=settings.db_connect_timeout_seconds
    )


def append_events(
    request_id: str,
    user_id: int | None,
    events: Iterable[AuditEvent],
    *,
    escalation_reason: str | None = None,
    connect: Callable = _connect,
) -> None:
    """Chain `events` onto the log, plus the `escalations` row if there is one.

    One transaction: a refusal is never in `escalations` without its audit trail, or
    the reverse. If this raises, /query fails with 503 — no answer leaves unaudited.
    """
    with connect() as conn, conn.transaction():
        append_in(conn, request_id, user_id, events, escalation_reason=escalation_reason)


def append_in(conn, request_id, user_id, events, *, escalation_reason=None) -> None:
    """The append itself, inside a transaction the caller owns — so a change and the
    audit row recording it commit or roll back together."""
    from psycopg.types.json import Jsonb

    conn.execute("SELECT pg_advisory_xact_lock(%s)", (_CHAIN_LOCK,))
    last = conn.execute("SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    prev = last[0] if last else None
    for event in events:
        payload = _canonical(event.payload)
        digest = row_hash(prev, request_id, event.event_type, user_id, payload, event.occurred_at)
        conn.execute(
            """
            INSERT INTO audit_log
                (request_id, event_type, user_id, payload, prev_hash, row_hash, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (request_id, event.event_type, user_id, Jsonb(payload), prev, digest,
             event.occurred_at),
        )
        prev = digest
    if escalation_reason:
        conn.execute(
            "INSERT INTO escalations (request_id, reason) VALUES (%s, %s)",
            (request_id, escalation_reason),
        )


def events_for(state: GraphState) -> list[AuditEvent]:
    """The request's trail, in the order the graph produced it.

    ponytail: derived from the final state by one terminal node, not written at each
    transition, so a multi-hop request records its last hop only (and the hop count).
    Wrap each node in build_graph if per-hop rows are ever needed.
    """
    now = datetime.now(UTC)

    def event(event_type, payload):
        return AuditEvent(event_type=event_type, payload=payload, occurred_at=now)

    trail = [event("query_received", {"query": state["query"], "route": state.get("route")})]

    if state.get("hop_count"):
        chunks = state.get("retrieved_chunks") or []
        trail.append(event("retrieval", {
            "hops": state["hop_count"],
            # Identity, not content: the log must not become a second copy of the corpus.
            "chunk_ids": [c.id for c in chunks],
            "document_ids": sorted({c.document_id for c in chunks}),
            "scores": [round(c.score, 4) for c in chunks],
        }))
    if state.get("sql_result") is not None:
        trail.append(event("sql_executed", {"result": state["sql_result"]}))
    if state.get("permission_conflicts"):
        trail.append(event("permission_conflict", {
            "conflicts": [c.model_dump(mode="json") for c in state["permission_conflicts"]],
        }))
    if state.get("draft_answer"):
        trail.append(event("draft_answer", {"text": state["draft_answer"]}))
    if state.get("verification") is not None:
        trail.append(event("verification", state["verification"].model_dump(mode="json")))

    # Written by nodes that know something the final state doesn't (Escalation's
    # reason, a cache hit), kept in their own order.
    trail.extend(state.get("audit_events") or [])

    trail.append(event("final_answer", {
        "escalated": bool(state.get("escalated")),
        "text": state.get("final_answer"),
        "citations": [c.model_dump(mode="json") for c in state.get("citations") or []],
    }))
    return trail


def audit_node(state: GraphState, connect: Callable = _connect) -> dict:
    trail = events_for(state)
    reason = next(
        (e.payload.get("reason") for e in trail if e.event_type == "escalation"), None
    )
    append_events(
        state["request_id"], state["user"].id, trail, escalation_reason=reason, connect=connect
    )
    return {"audit_events": trail}


def read_request(request_id: str, connect: Callable = _connect) -> list[AuditEvent]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT event_type, payload, created_at FROM audit_log "
            "WHERE request_id = %s ORDER BY id",
            (request_id,),
        ).fetchall()
    return [AuditEvent(event_type=t, payload=p, occurred_at=c) for t, p, c in rows]


@dataclass
class ChainReport:
    ok: bool
    rows_checked: int
    first_bad_id: int | None = None
    problem: str | None = None


def verify_rows(rows: Iterable[tuple]) -> ChainReport:
    """Walk (id, request_id, event_type, user_id, payload, prev_hash, row_hash, created_at)
    in id order. Stops at the first row that doesn't recompute or doesn't link."""
    expected_prev = None
    checked = 0
    for row_id, request_id, event_type, user_id, payload, prev, digest, created_at in rows:
        checked += 1
        if prev != expected_prev:
            return ChainReport(False, checked, row_id,
                               "prev_hash does not match the previous row — a row was removed or inserted")
        if row_hash(prev, request_id, event_type, user_id, payload, created_at) != digest:
            return ChainReport(False, checked, row_id,
                               "row_hash does not recompute — this row was edited")
        expected_prev = digest
    return ChainReport(True, checked)


def verify_audit_chain(connect: Callable = _connect) -> ChainReport:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, request_id, event_type, user_id, payload, prev_hash, row_hash, created_at "
            "FROM audit_log ORDER BY id"
        )
        return verify_rows(rows)


def main(argv: list[str]) -> int:
    if argv[1:] != ["verify"]:
        print("usage: python -m src.agents.audit verify")
        return 2
    report = verify_audit_chain()
    if report.ok:
        print(f"OK — {report.rows_checked} rows, chain intact")
        return 0
    print(f"TAMPERED — row {report.first_bad_id}: {report.problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
