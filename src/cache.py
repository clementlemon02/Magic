"""Permission-aware semantic answer cache.

A plain semantic cache is the cheapest latency win in RAG and, in a system with
per-document access control, a disclosure hole: it will happily serve one caller an
answer assembled from documents another caller could read.

Keying by user instead throws the benefit away. This keys by *permission
fingerprint* — the caller's ACL tags plus the exact set of grants currently live in
the `permissions` table — so callers with identical access share entries and nobody
else does. Revoking a grant changes the fingerprint, which invalidates the affected
entries with no extra bookkeeping: the same property §1 already relies on, that
authorisation is re-derived rather than remembered.

The corpus is part of the fingerprint too. A sync (src/ingestion/sync.py) bumps
`corpus_state.epoch` whenever it changes a document, a chunk or a grant, so an answer
cached before an edit stops matching afterwards. It is read from the database rather than
cleared in-process because the sync may run in another process (`scripts.sync_sources`,
a second worker) and nothing else would tell this one. `ttl_seconds` is the backstop for
any change the epoch does not see.

An entry is filed under the fingerprint taken BEFORE the answer was computed, which `/query`
passes back to `put`. Re-reading it after would file an answer built from the old corpus
under the new one: a request that straddles a sync, or a revoke, would then serve stale
evidence to everyone who shares the new fingerprint — including evidence they may no longer
read — until it aged out.

ponytail: in-process dict, so it is per-worker and empty after a restart. That is
the right size for a demo; move to Redis or a pgvector table if it has to survive
a deploy or be shared across workers.
"""

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime

from src.graph.state import AskerResponse, UserContext

FINGERPRINT_SQL = """
    SELECT source_platform, source_ref
    FROM permissions
    WHERE user_id = %(user_id)s AND revoked_at IS NULL
    ORDER BY source_platform, source_ref
"""

EPOCH_SQL = "SELECT epoch FROM corpus_state WHERE id = 1"


@dataclass
class CacheEntry:
    vector: list[float]
    fingerprint: str
    response: AskerResponse
    stored_at: datetime
    # The route that produced this answer, replayed into the audit trail on a hit.
    # Without it a hit is logged with initial_state's `rag` placeholder, so a cached
    # sql or decline request was recorded as a rag one — wrong rows in a log whose
    # whole selling point is that it can be trusted.
    route: str | None = None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _psycopg_execute(sql: str, params: dict) -> list[tuple]:
    import psycopg

    from src.config import get_settings

    settings = get_settings()
    with psycopg.connect(
        settings.database_url, connect_timeout=settings.db_connect_timeout_seconds
    ) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


class PermissionAwareCache:
    """Semantic cache partitioned by what the caller is currently allowed to read."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.93,
        max_entries: int = 256,
        ttl_seconds: float | None = None,
        embeddings=None,
        execute=None,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._embeddings = embeddings
        self._execute = execute or _psycopg_execute
        self._entries: list[CacheEntry] = []
        self.hits = 0
        self.misses = 0

    def _embed(self, query: str) -> list[float]:
        if self._embeddings is None:
            from src.llm.factory import get_embeddings

            self._embeddings = get_embeddings()
        return self._embeddings.embed_query(query)

    def fingerprint(self, user: UserContext) -> str:
        """A stable digest of everything this caller may currently read.

        Covers the ACL tags AND the live grant rows, because two callers can share a
        role and still differ by an individual grant, and the corpus epoch, so a sync
        invalidates what it made stale. Two indexed queries — far less than the
        retrieval and three model calls a miss would run.
        """
        grants = self._execute(FINGERPRINT_SQL, {"user_id": user.id})
        epoch = self._execute(EPOCH_SQL, {})
        material = "|".join(sorted(user.acl_tags()))
        material += "||" + "|".join(f"{platform}:{ref}" for platform, ref in grants)
        material += "||epoch:" + (str(epoch[0][0]) if epoch else "none")
        return hashlib.sha256(material.encode()).hexdigest()

    def get(self, query: str, user: UserContext) -> AskerResponse | None:
        """The cached answer, or None. `lookup` when you also need how it was routed."""
        entry = self.lookup(query, user)
        return entry.response if entry else None

    def lookup(
        self, query: str, user: UserContext, *, fingerprint: str | None = None
    ) -> CacheEntry | None:
        digest = fingerprint or self.fingerprint(user)
        now = datetime.now(UTC)
        candidates = [
            e for e in self._entries
            if e.fingerprint == digest
            and (self.ttl_seconds is None or (now - e.stored_at).total_seconds() < self.ttl_seconds)
        ]
        if not candidates:
            self.misses += 1
            return None

        vector = self._embed(query)
        best, score = None, 0.0
        for entry in candidates:
            similarity = _cosine(vector, entry.vector)
            if similarity > score:
                best, score = entry, similarity

        if best is not None and score >= self.similarity_threshold:
            self.hits += 1
            return best
        self.misses += 1
        return None

    def put(
        self,
        query: str,
        user: UserContext,
        response: AskerResponse,
        route: str | None = None,
        *,
        fingerprint: str | None = None,
    ) -> None:
        self._entries.append(
            CacheEntry(
                vector=self._embed(query),
                fingerprint=fingerprint or self.fingerprint(user),
                response=response,
                stored_at=datetime.now(UTC),
                route=route,
            )
        )
        if len(self._entries) > self.max_entries:
            del self._entries[: len(self._entries) - self.max_entries]

    def clear(self) -> None:
        self._entries.clear()
        self.hits = self.misses = 0
