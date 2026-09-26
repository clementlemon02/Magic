"""Rebuild the demo corpus, grants and transactions from scratch.

    .venv/bin/python -m scripts.seed_demo

Run once after `docker compose up -d` on any machine that will present. Needs Ollama
with mxbai-embed-large pulled. Safe to re-run: it replaces documents, chunks,
permissions and transactions, and leaves users and audit_log alone. The audit
chain is append-only, and reseeding must not be a way to wipe it.

Grants come from each connector's own `check_access`, evaluated here, once, for
every user in the database. That stands in for the nightly sync from each
platform's ACL that a real deployment would run. It is never consulted at
request time (§1): retrieval joins the `permissions` rows this writes.
"""

import random
import sys
from datetime import UTC, datetime, timedelta

import psycopg

from src.api.main import load_user
from src.config import get_settings
from src.connectors import connectors
from src.ingestion.service import ingest_connector
from src.llm.factory import get_embeddings

CONNECTORS = tuple(connectors().values())


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
    embeddings = get_embeddings()

    with psycopg.connect(settings.database_url) as conn:
        user_ids = [r[0] for r in conn.execute("SELECT id FROM users ORDER BY id")]
        if not user_ids:
            print("no users — apply scripts/seed_users.sql first")
            return 1
        conn.execute("TRUNCATE documents, document_chunks, permissions, transactions RESTART IDENTITY")

        docs = chunks = 0
        for connector in CONNECTORS:
            # ingest_connector commits, so a failure part-way leaves earlier connectors in.
            summary = ingest_connector(connector, conn, embeddings)
            docs += summary.documents_ingested
            chunks += summary.chunks_ingested

        users = [load_user(uid) for uid in user_ids]
        grants = []
        for connector in CONNECTORS:
            for item in connector.list_items():
                for user in users:
                    if connector.check_access(user, item):
                        grants.append((user.id, item.platform, item.source_ref))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO permissions (user_id, source_platform, source_ref) VALUES (%s, %s, %s)",
                grants,
            )
            cur.executemany(
                "INSERT INTO transactions (account_dept, amount, flagged_aml, occurred_at, acl_tags) "
                "VALUES (%s, %s, %s, %s, %s)",
                list(_transactions(random.Random(42))),
            )
        conn.commit()

    print(f"{docs} documents, {chunks} chunks, {len(grants)} grants, 600 transactions")
    for user in users:
        mine = sorted(f"{p}:{r}" for uid, p, r in grants if uid == user.id)
        print(f"  user {user.id} ({user.role}): {', '.join(mine)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
