"""Rebuild the demo corpus, grants and transactions from scratch.

    .venv/bin/python -m scripts.seed_demo

Run once after `docker compose up -d` on any machine that will present. Needs Ollama
with mxbai-embed-large pulled. Safe to re-run: it replaces documents, chunks,
permissions and transactions, and leaves users and audit_log alone. The audit
chain is append-only, and reseeding must not be a way to wipe it.

Documents and grants are loaded by src/ingestion/sync.py, the same code the server runs
on its schedule, against tables this empties first. Each grant comes from the source
connector's own `check_access`, evaluated for every user in the database. It is never
consulted at request time (§1): retrieval joins the `permissions` rows this writes.
After the first load, `python -m scripts.sync_sources` brings the mirror up to date
without emptying anything.
"""

import random
import sys
from datetime import UTC, datetime, timedelta

import psycopg

from src.config import get_settings
from src.db.migrate import migrate
from src.ingestion.sync import sync_sources
from src.llm.factory import get_embeddings


def _transactions(rng: random.Random):
    """Deterministic, so a demo number is the same number at every rehearsal."""
    start = datetime(2026, 6, 1, tzinfo=UTC)
    for _ in range(600):
        dept = rng.choice(("support", "compliance"))
        yield (
            dept,
            round(rng.uniform(20, 15_000), 2),
            rng.random() < 0.04,  # ~4% flagged for AML
            start + timedelta(minutes=rng.randrange(0, 122 * 24 * 60)),  # Jun-Sep
            # Visible to the owning department; Compliance oversees every account.
            sorted({dept, "compliance"}),
        )


def main() -> int:
    settings = get_settings()
    migrate()

    with psycopg.connect(settings.database_url) as conn:
        if not conn.execute("SELECT 1 FROM users LIMIT 1").fetchone():
            print("no users — apply scripts/seed_users.sql first")
            return 1
        conn.execute("TRUNCATE documents, document_chunks, permissions, transactions RESTART IDENTITY")
        # Committed BEFORE the sync, which opens connections of its own: TRUNCATE holds an
        # exclusive lock until it commits, and the sync would wait on it forever.
        conn.commit()

    reports = sync_sources(embeddings=get_embeddings(), trigger="seed")
    failed = [r for r in reports if r.error]
    if failed:
        for r in failed:
            print(f"{r.platform}: {r.error}")
        return 1

    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO transactions (account_dept, amount, flagged_aml, occurred_at, acl_tags) "
                "VALUES (%s, %s, %s, %s, %s)",
                list(_transactions(random.Random(42))),
            )
        conn.commit()
        docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT count(*) FROM document_chunks").fetchone()[0]
        grants = conn.execute(
            "SELECT p.user_id, r.name, p.source_platform || ':' || p.source_ref "
            "FROM permissions p JOIN users u ON u.id = p.user_id JOIN roles r ON r.id = u.role_id "
            "WHERE p.revoked_at IS NULL ORDER BY p.user_id, 3"
        ).fetchall()

    print(f"{docs} documents, {chunks} chunks, {len(grants)} grants, 600 transactions")
    for user_id, role in sorted({(g[0], g[1]) for g in grants}):
        mine = ", ".join(g[2] for g in grants if g[0] == user_id)
        print(f"  user {user_id} ({role}): {mine}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
