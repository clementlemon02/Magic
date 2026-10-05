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

    Two kinds of row. What the nodes wrote as they ran — one `node_transition` each,
    plus whatever a node recorded itself — comes first, in the order it happened and
    with its own timestamps. The summary of the finished request follows, built here
    from final state, so those rows all carry the same closing time.

    ponytail: the summary still describes the LAST hop's chunks and verdict. The rows
    above it say a hop happened and how long it took, not what that hop retrieved;
    put that detail in retrieval's own event if an auditor ever needs it.
    """
    now = datetime.now(UTC)

    def event(event_type, payload):
        return AuditEvent(event_type=event_type, payload=payload, occurred_at=now)

    # Written by the nodes themselves, already in order: one per transition, plus
    # Escalation's reason, a cache hit, a source recheck.
    recorded = list(state.get("audit_events") or [])

    # The question arrived before any node ran, so it takes the first node's time
    # rather than `now`. Given `now`, the opening row of the trail carried the
    # LATEST timestamp in it, which reads like the request began after it ended.
    began = recorded[0].occurred_at if recorded else now
    trail = [
        AuditEvent(
            event_type="query_received",
            payload={"query": state["query"], "route": state.get("route")},
            occurred_at=began,
        )
    ]
    trail.extend(recorded)

    if state.get("hop_count"):
        chunks = state.get("retrieved_chunks") or []
        trail.append(event("retrieval", {
            "hops": state["hop_count"],
            # Identity, not content: the log must not become a second copy of the corpus.
            "chunk_ids": [c.id for c in chunks],
            "document_ids": sorted({c.document_id for c in chunks}),
            # By name as well as by id. A document id means a row in the corpus AS IT WAS:
            # reseeding restarts the sequence, and a sync that removes and re-adds a document
            # gives it a new one, so an old id can point at a different document. The name is
            # what lets an officer ask "who was shown anything from this space" a month later.
            "sources": list(dict.fromkeys(
                f"{c.citation.source_platform}:{c.citation.source_ref}" for c in chunks
            )),
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


# Every document an audited request NAMES, by "platform:ref", with the request it belongs to.
# Four ways a request names one, and an officer asking "what did this person get near" wants
# all four: what the answer was built from (citations), what was retrieved for it (new rows
# only; older ones carry ids, which are not stable names), what was refused because it
# outranked the permitted evidence (permission_conflict) and what the source itself denied
# at question time (source_recheck_denied). Matching on EXTRACTED refs, never on the payload
# text: an answer that merely mentions a document must not count as having touched it.
NAMED_DOCUMENTS = """
    SELECT e.request_id, (item->>'source_platform') || ':' || (item->>'source_ref') AS ref
    FROM audit_log AS e
    CROSS JOIN LATERAL jsonb_array_elements(e.payload->'citations') AS item
    WHERE e.event_type = 'final_answer' AND e.created_at >= %(since)s
    UNION ALL
    SELECT e.request_id, (item->>'source_platform') || ':' || (item->>'source_ref')
    FROM audit_log AS e
    CROSS JOIN LATERAL jsonb_array_elements(e.payload->'conflicts') AS item
    WHERE e.event_type = 'permission_conflict' AND e.created_at >= %(since)s
    UNION ALL
    SELECT e.request_id, (item->>'source_platform') || ':' || (item->>'source_ref')
    FROM audit_log AS e
    CROSS JOIN LATERAL jsonb_array_elements(e.payload->'items') AS item
    WHERE e.event_type = 'source_recheck_denied' AND e.created_at >= %(since)s
    UNION ALL
    SELECT e.request_id, source
    FROM audit_log AS e
    CROSS JOIN LATERAL jsonb_array_elements_text(e.payload->'sources') AS source
    WHERE e.event_type = 'retrieval' AND e.created_at >= %(since)s
"""

RECENT_REQUESTS = f"""
    WITH named AS ({NAMED_DOCUMENTS})
    SELECT question.request_id,
           question.created_at,
           question.user_id,
           question.payload->>'query' AS query,
           question.payload->>'route' AS route,
           coalesce((final.payload->>'escalated')::boolean, false) AS escalated,
           escalation.payload->>'reason' AS reason,
           asker.name AS user_name
    FROM audit_log AS question
    LEFT JOIN audit_log AS final
      ON final.request_id = question.request_id AND final.event_type = 'final_answer'
    LEFT JOIN audit_log AS escalation
      ON escalation.request_id = question.request_id AND escalation.event_type = 'escalation'
    LEFT JOIN users AS asker ON asker.id = question.user_id
    WHERE question.event_type = 'query_received' AND question.created_at >= %(since)s
      -- Filters applied HERE, not in the page. 306 requests over 7 days against a
      -- 100-row window: a page that narrowed what it had already fetched would show
      -- "12 refusals" when the window happened to hold 12 of 122, which is the same
      -- lie as the count that used to read "100 requests".
      AND (%(outcome)s = 'all'
           OR (%(outcome)s = 'refused'  AND escalation.request_id IS NOT NULL)
           OR (%(outcome)s = 'answered' AND escalation.request_id IS NULL
               AND question.payload->>'route' IS DISTINCT FROM 'decline')
           OR (%(outcome)s = 'declined' AND question.payload->>'route' = 'decline'))
      AND (%(q)s = '' OR question.payload->>'query' ILIKE %(like)s)
      -- Whose requests: an id, or a fragment of a name or an email. All digits is an id and
      -- nothing else, or "2" would also find anyone with a 2 in their address.
      AND (%(user)s = ''
           OR asker.id = %(user_id)s::int
           OR (%(user_id)s::int IS NULL
               AND (asker.name ILIKE %(user_like)s OR asker.email ILIKE %(user_like)s)))
      -- Which documents: anything the request named, by "platform:ref" fragment.
      AND (%(document)s = ''
           OR question.request_id IN (SELECT request_id FROM named WHERE ref ILIKE %(document_like)s))
    ORDER BY question.created_at DESC
    LIMIT %(limit)s
"""

# What an officer can narrow to. `declined` is worth its own bucket rather than
# living under `answered`: 50 of those 306 were greetings and "write me a poem",
# which is noise on a screen whose job is refusals.
OUTCOMES = ("all", "refused", "answered", "declined")


def _like(text: str) -> str:
    """A LIKE pattern for "contains `text`", with the caller's own % and _ taken literally.

    Bound by the driver like everything else; the surrounding wildcards are ours. Unescaped,
    searching "50%" would match every row, which on this page reads as "your filter found
    everything" rather than "your filter did nothing".
    """
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def recent_requests(
    since: datetime,
    *,
    limit: int = 100,
    outcome: str = "all",
    q: str = "",
    user: str = "",
    document: str = "",
    connect: Callable = _connect,
) -> list[dict[str, Any]]:
    """Requests in the window, newest first, with how each ended.

    The asker-facing routes cannot list these: a request id is the key to an
    explanation, so browsing them is an officer's job (src/api/compliance.py).
    The question text comes along because an officer working through refusals needs
    to see what was asked, not a column of UUIDs.

    The brief's inquiry is "everything user jdoe accessed related to the payment-gateway
    space in the last 30 days", which is these filters together: `user` (an id, or part of a
    name or email), `document` (part of "platform:ref", so `confluence:SUPPORT/` is a whole
    space) and the window. They narrow in SQL, before the limit, so a filtered list is every
    match in the window rather than the matches that happened to fall inside the newest
    hundred. An unknown outcome falls back to 'all' — a typo must not silently hide rows
    from an audit surface.
    """
    if outcome not in OUTCOMES:
        outcome = "all"
    text, who, doc = (q or "").strip(), (user or "").strip(), (document or "").strip()
    with connect() as conn:
        rows = conn.execute(RECENT_REQUESTS, {
            "since": since, "limit": limit, "outcome": outcome,
            "q": text, "like": _like(text),
            "user": who, "user_id": int(who) if who.isdigit() else None, "user_like": _like(who),
            "document": doc, "document_like": _like(doc),
        }).fetchall()
    return [
        {
            "request_id": str(request_id),
            "at": at,
            "user_id": user_id,
            "user_name": user_name,
            "query": query or "",
            "route": route,
            "escalated": bool(escalated),
            "reason": reason,
        }
        for request_id, at, user_id, query, route, escalated, reason, user_name in rows
    ]


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
