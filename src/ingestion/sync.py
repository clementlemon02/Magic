"""Reconcile the mirror with the sources: documents, chunks, and who may read them.

Ingestion used to be a batch. `scripts.seed_demo` emptied the tables and loaded everything
again, so a page edited in Confluence, a channel someone left or a file deleted from Drive
reached answers only when somebody remembered to run it, and `ingest_connector` is
insert-only, so running it twice would have doubled the corpus.

This is the other half of CLAUDE.md §1's "mirror" caveat. The query-time recheck
(src/connectors/__init__.py) covers RESTRICTED documents between syncs; this keeps the whole
mirror inside a bound of one SYNC_INTERVAL_MINUTES plus the time a sync takes, rather than
"whenever someone reseeds".

Per platform, in ONE transaction:

  1. take a per-platform advisory lock, or skip: two syncs of one source must not race
  2. list the source as it is NOW, and the mirror as it is held
  3. plan: add / rewrite (text changed: re-chunk, re-embed) / retag (same text, different
     title, department, sensitivity or ACL: updated in place, nothing re-embedded) / remove
  4. embed only what the plan says changed
  5. apply it, then reconcile grants against the source's own `check_access`
  6. bump `corpus_state.epoch`, which the answer cache keys on (src/cache.py), so no cached
     answer from before the change is served after it, in this process or any other
  7. one `source_syncs` row, and one `source_sync` audit row when something changed

Any failure rolls that platform back: the mirror is exactly as it was and the failure is
recorded. A source that cannot be listed is therefore never read as "everything was deleted".

GRANTS. The source is authoritative for who may read an item, so a live grant the source no
longer backs is revoked, marked `revoked_by = 'source'`. The other direction is asymmetric on
purpose: a grant the source backs is restored only if a SYNC took it away. A revoke an officer
made through /admin/permissions/revoke stays revoked; re-granting it ten minutes later would
undo a deliberate decision. Neither direction can widen access, because §1 needs tag overlap
AND a live grant, and a grant alone is never enough.

ponytail: the whole platform runs in one transaction, so it sits idle-in-transaction while the
model embeds. Fine for hundreds of documents; past that, embed first and re-read under the lock.
Change detection is by content hash. A row with no hash (written before the column existed) is
treated as changed, so the first sync after an upgrade re-embeds once and can never miss an edit.
"""

import hashlib
import logging
import uuid
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel

from src.agents import audit
from src.connectors.base import SourceConnector, SourceItem
from src.graph.state import AuditEvent, UserContext
from src.ingestion.service import _insert_chunk, chunk_text, normalize_acl_tags

logger = logging.getLogger(__name__)

# With the platform name hashed in, one lock per source. Arbitrary but fixed.
SYNC_LOCK = 0x53594E43  # "SYNC"
SOURCE = "source"  # permissions.revoked_by for a grant a sync took away
# The audit payload names who lost access; a source that re-permissions thousands of items
# would otherwise write a payload nobody can read. The count is always exact.
AUDIT_PAIR_CAP = 200

HELD_SQL = """
    SELECT id, source_ref, content_hash, title, dept, sensitivity, acl_tags
    FROM documents WHERE source_platform = %s
"""
INSERT_DOCUMENT = """
    INSERT INTO documents (title, source_platform, source_ref, dept, sensitivity, acl_tags,
                           content_hash, source_updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
"""
UPDATE_DOCUMENT = """
    UPDATE documents SET title = %s, dept = %s, sensitivity = %s, acl_tags = %s,
                         content_hash = %s, source_updated_at = %s, synced_at = now()
    WHERE id = %s
"""
USERS_SQL = """
    SELECT u.id, r.name, u.dept, r.clearance_level
    FROM users AS u JOIN roles AS r ON r.id = u.role_id ORDER BY u.id
"""
# The newest row per (user, item): whether it is live, and if not, who revoked it.
LATEST_GRANTS_SQL = """
    SELECT DISTINCT ON (user_id, source_ref) user_id, source_ref, revoked_at IS NULL, revoked_by
    FROM permissions WHERE source_platform = %s
    ORDER BY user_id, source_ref, id DESC
"""
GRANT_SQL = "INSERT INTO permissions (user_id, source_platform, source_ref) VALUES (%s, %s, %s)"
REVOKE_SQL = f"""
    UPDATE permissions SET revoked_at = now(), revoked_by = '{SOURCE}'
    WHERE user_id = %s AND source_platform = %s AND source_ref = %s AND revoked_at IS NULL
"""
BUMP_EPOCH_SQL = """
    INSERT INTO corpus_state (id, epoch, changed_at) VALUES (1, 1, now())
    ON CONFLICT (id) DO UPDATE SET epoch = corpus_state.epoch + 1, changed_at = now()
"""
RECORD_SQL = """
    INSERT INTO source_syncs (source_platform, trigger, started_at, finished_at, added, updated,
                              removed, unchanged, grants_added, grants_revoked, error)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


class SyncReport(BaseModel):
    """What one platform's sync did. Counts only, so it is safe on an officer's screen."""

    platform: str
    trigger: str
    started_at: datetime
    finished_at: datetime
    added: int = 0
    updated: int = 0  # text rewritten, or title/department/sensitivity/ACL changed
    removed: int = 0
    unchanged: int = 0
    grants_added: int = 0
    grants_revoked: int = 0
    error: str | None = None
    skipped: bool = False  # another sync of this source was already running


@dataclass(frozen=True)
class Wanted:
    """One item as the source describes it now, ready to compare with the mirror."""

    item: SourceItem
    acl_tags: list[str]
    chunks: list[str]
    content_hash: str


@dataclass(frozen=True)
class Held:
    """One document as the mirror holds it. Field order is HELD_SQL's column order."""

    id: int
    source_ref: str
    content_hash: str | None
    title: str
    dept: str
    sensitivity: str
    acl_tags: list[str]


@dataclass
class Plan:
    add: list[Wanted] = field(default_factory=list)
    rewrite: list[tuple[Held, Wanted]] = field(default_factory=list)
    retag: list[tuple[Held, Wanted]] = field(default_factory=list)
    remove: list[Held] = field(default_factory=list)
    unchanged: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.add or self.rewrite or self.retag or self.remove)


def _digest(text: str) -> str:
    """Whitespace-normalised, as chunk_text is, so reflowing a page is not an edit."""
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def _wanted(connector: SourceConnector) -> dict[str, Wanted]:
    wanted = {}
    for item in connector.list_items():
        chunks = chunk_text(item.content)
        if not chunks:
            continue  # ingest_connector skips these too; one that went empty is removed
        wanted[item.source_ref] = Wanted(
            item, normalize_acl_tags(connector.permissions_for(item)), chunks, _digest(item.content)
        )
    return wanted


def make_plan(wanted: Mapping[str, Wanted], held: Mapping[str, Held]) -> Plan:
    """Pure: what must change in the mirror to match the source. Keyed by source_ref."""
    plan = Plan()
    for ref, w in wanted.items():
        h = held.get(ref)
        if h is None:
            plan.add.append(w)
        elif h.content_hash != w.content_hash:
            plan.rewrite.append((h, w))
        elif (h.title, h.dept, h.sensitivity, sorted(h.acl_tags)) != (
            w.item.title, w.item.dept, w.item.sensitivity, sorted(w.acl_tags)
        ):
            plan.retag.append((h, w))
        else:
            plan.unchanged += 1
    plan.remove = [h for ref, h in held.items() if ref not in wanted]
    return plan


def grant_changes(
    entitled: Collection[tuple[int, str]],
    latest: Mapping[tuple[int, str], tuple[bool, str | None]],
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Pure: (grants to add, grants to revoke) for (user_id, source_ref) pairs.

    `latest` maps a pair to its newest row, (is_live, revoked_by). Add what the source backs
    and no row exists for, or a sync revoked. Never add over a revoke an officer made.
    Revoke every live grant the source no longer backs, including on items it no longer has.
    """
    grant = sorted(
        pair for pair in entitled
        if pair not in latest or (not latest[pair][0] and latest[pair][1] == SOURCE)
    )
    revoke = sorted(pair for pair, (live, _) in latest.items() if live and pair not in entitled)
    return grant, revoke


def _embed(embeddings, w: Wanted) -> list[list[float]]:
    vectors = embeddings.embed_documents(w.chunks)
    if len(vectors) != len(w.chunks):
        raise RuntimeError("Embedding provider returned a vector count that does not match chunk count")
    return vectors


def _store(conn, document_id: int, w: Wanted, vectors: list[list[float]]) -> None:
    for content, embedding in zip(w.chunks, vectors, strict=True):
        _insert_chunk(conn, document_id=document_id, content=content, embedding=embedding, acl_tags=w.acl_tags)


def _fields(w: Wanted) -> tuple:
    return (w.item.title, w.item.dept, w.item.sensitivity, w.acl_tags, w.content_hash, w.item.updated_at)


def _pairs(pairs: list[tuple[int, str]]) -> list[dict]:
    return [{"user_id": u, "source_ref": r} for u, r in pairs[:AUDIT_PAIR_CAP]]


def _sync_platform(conn, connector, embeddings, trigger: str, user_id: int | None) -> SyncReport:
    platform = connector.platform
    started = datetime.now(UTC)
    locked = conn.execute(
        "SELECT pg_try_advisory_xact_lock(%s, hashtext(%s))", (SYNC_LOCK, platform)
    ).fetchone()[0]
    if not locked:
        return SyncReport(platform=platform, trigger=trigger, started_at=started,
                          finished_at=started, skipped=True)

    wanted = _wanted(connector)
    held = {row[1]: Held(*row) for row in conn.execute(HELD_SQL, (platform,)).fetchall()}
    plan = make_plan(wanted, held)
    vectors = {w.item.source_ref: _embed(embeddings, w)
               for w in (*plan.add, *(w for _, w in plan.rewrite))}

    for w in plan.add:
        document_id = conn.execute(
            INSERT_DOCUMENT, (w.item.title, platform, w.item.source_ref, w.item.dept,
                              w.item.sensitivity, w.acl_tags, w.content_hash, w.item.updated_at),
        ).fetchone()[0]
        _store(conn, document_id, w, vectors[w.item.source_ref])
    for h, w in plan.rewrite:
        # The document keeps its id, so what already points at it (citations in the audit
        # trail) still resolves; only its text and embeddings are replaced.
        conn.execute("DELETE FROM document_chunks WHERE document_id = %s", (h.id,))
        conn.execute(UPDATE_DOCUMENT, (*_fields(w), h.id))
        _store(conn, h.id, w, vectors[w.item.source_ref])
    for h, w in plan.retag:
        conn.execute(UPDATE_DOCUMENT, (*_fields(w), h.id))
        conn.execute("UPDATE document_chunks SET acl_tags = %s WHERE document_id = %s",
                     (w.acl_tags, h.id))
    if plan.remove:
        conn.execute("DELETE FROM documents WHERE id = ANY(%s)", ([h.id for h in plan.remove],))

    users = [UserContext(id=i, role=r, dept=d, clearance_level=c)
             for i, r, d, c in conn.execute(USERS_SQL).fetchall()]
    entitled = {(u.id, ref) for ref, w in wanted.items() for u in users
                if connector.check_access(u, w.item)}
    latest = {(u, ref): (live, by)
              for u, ref, live, by in conn.execute(LATEST_GRANTS_SQL, (platform,)).fetchall()}
    to_grant, to_revoke = grant_changes(entitled, latest)
    with conn.cursor() as cur:
        cur.executemany(GRANT_SQL, [(u, platform, ref) for u, ref in to_grant])
        cur.executemany(REVOKE_SQL, [(u, platform, ref) for u, ref in to_revoke])

    finished = datetime.now(UTC)
    changed = plan.changed or bool(to_grant or to_revoke)
    report = SyncReport(
        platform=platform, trigger=trigger, started_at=started, finished_at=finished,
        added=len(plan.add), updated=len(plan.rewrite) + len(plan.retag),
        removed=len(plan.remove), unchanged=plan.unchanged,
        grants_added=len(to_grant), grants_revoked=len(to_revoke),
    )
    if changed:
        conn.execute(BUMP_EPOCH_SQL)
    # An officer pressing the button is on the record even when nothing moved, like every
    # other officer action (src/api/compliance.py); a quiet scheduled run is not.
    if changed or trigger == "manual":
        audit.append_in(conn, str(uuid.uuid4()), user_id, [AuditEvent(
            event_type="source_sync", occurred_at=finished,
            payload={
                "platform": platform, "trigger": trigger,
                "added": [w.item.source_ref for w in plan.add],
                "rewritten": [w.item.source_ref for _, w in plan.rewrite],
                "retagged": [w.item.source_ref for _, w in plan.retag],
                "removed": [h.source_ref for h in plan.remove],
                "grants_added": len(to_grant), "grants_revoked": len(to_revoke),
                "revoked": _pairs(to_revoke), "granted": _pairs(to_grant),
            },
        )])
    _record(conn, report)
    return report


LAST_SYNCED_SQL = """
    SELECT source_platform, max(finished_at) FROM source_syncs WHERE error IS NULL
    GROUP BY source_platform
"""


def last_synced(connect: Callable | None = None) -> dict[str, datetime]:
    """When each source last synced successfully, for the freshness stamp on a citation.

    Best effort: an annotation must never cost somebody their answer, so any failure,
    including a database that has not been migrated yet, is an empty answer.
    """
    try:
        with (connect or audit._connect)() as conn:
            return dict(conn.execute(LAST_SYNCED_SQL).fetchall())
    except Exception:  # noqa: BLE001
        return {}


def _record(conn, report: SyncReport) -> None:
    conn.execute(RECORD_SQL, (
        report.platform, report.trigger, report.started_at, report.finished_at, report.added,
        report.updated, report.removed, report.unchanged, report.grants_added,
        report.grants_revoked, report.error,
    ))


def sync_sources(
    *,
    connect: Callable | None = None,
    embeddings=None,
    sources: Mapping[str, SourceConnector] | None = None,
    platforms: Collection[str] | None = None,
    trigger: str = "manual",
    user_id: int | None = None,
) -> list[SyncReport]:
    """Reconcile each source with the mirror. Never raises for a source: it reports.

    `connect` returns a context-managed connection, as src/agents/audit._connect does, so a
    test can hand every call one connection inside a transaction it rolls back.
    `user_id` is the officer who asked, for a manual run, and None for the schedule.
    """
    if connect is None:
        connect = audit._connect
    if embeddings is None:
        from src.llm.factory import get_embeddings

        embeddings = get_embeddings()
    if sources is None:
        from src.connectors import connectors

        sources = connectors()

    reports = []
    for platform, connector in sources.items():
        if platforms and platform not in platforms:
            continue
        started = datetime.now(UTC)
        try:
            with connect() as conn, conn.transaction():
                report = _sync_platform(conn, connector, embeddings, trigger, user_id)
        except Exception as error:  # noqa: BLE001 - a failing source must not stop the others
            logger.warning("sync of %s failed: %s", platform, error, exc_info=error)
            report = SyncReport(
                platform=platform, trigger=trigger, started_at=started,
                finished_at=datetime.now(UTC), error=f"{type(error).__name__}: {error}"[:300],
            )
            try:
                with connect() as conn, conn.transaction():
                    _record(conn, report)
            except Exception as record_error:  # noqa: BLE001 - e.g. the database itself is down
                logger.warning("could not record the failed sync of %s: %s", platform, record_error)
        reports.append(report)
    return reports
