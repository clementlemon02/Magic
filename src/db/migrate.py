"""Bring an existing database up to the current schema.

`schema.sql` only runs on the first boot of an empty volume, so a database created before a
column existed would otherwise need a hand-written ALTER. Every statement here is additive
and idempotent: running them twice, or against a database that is already current, does
nothing. Nothing is dropped, renamed or rewritten.

    python -m src.db.migrate

Keep this in step with schema.sql: that file is what a fresh database gets, this is how an
old one catches up.
"""

import sys

STATEMENTS = (
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_updated_at TIMESTAMPTZ",
    # No default on the ADD: existing rows would all be stamped with the migration time, which
    # would claim they had just been synced. NULL says "not known"; new rows get the default.
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS synced_at TIMESTAMPTZ",
    "ALTER TABLE documents ALTER COLUMN synced_at SET DEFAULT now()",
    "CREATE UNIQUE INDEX IF NOT EXISTS documents_source_key ON documents (source_platform, source_ref)",
    "ALTER TABLE permissions ADD COLUMN IF NOT EXISTS revoked_by TEXT",
    """
    CREATE TABLE IF NOT EXISTS source_syncs (
        id              BIGSERIAL PRIMARY KEY,
        source_platform TEXT NOT NULL,
        trigger         TEXT NOT NULL,
        started_at      TIMESTAMPTZ NOT NULL,
        finished_at     TIMESTAMPTZ NOT NULL,
        added           INTEGER NOT NULL DEFAULT 0,
        updated         INTEGER NOT NULL DEFAULT 0,
        removed         INTEGER NOT NULL DEFAULT 0,
        unchanged       INTEGER NOT NULL DEFAULT 0,
        grants_added    INTEGER NOT NULL DEFAULT 0,
        grants_revoked  INTEGER NOT NULL DEFAULT 0,
        error           TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS source_syncs_platform_finished ON source_syncs (source_platform, finished_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS corpus_state (
        id          SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
        epoch       BIGINT NOT NULL DEFAULT 0,
        changed_at  TIMESTAMPTZ
    )
    """,
    "INSERT INTO corpus_state (id, epoch) VALUES (1, 0) ON CONFLICT (id) DO NOTHING",
)


def ensure_schema(conn) -> None:
    """Apply every statement on `conn`. The caller owns the transaction."""
    import psycopg

    for statement in STATEMENTS:
        try:
            conn.execute(statement)
        except psycopg.errors.UniqueViolation as error:
            # Only the unique index on (source_platform, source_ref) can say this: the table
            # already holds the same document twice, which only an ingest run without a
            # TRUNCATE produces. Everything rolls back; say what to do instead of a raw error.
            raise RuntimeError(
                "documents already holds duplicate (source_platform, source_ref) rows, so it "
                "cannot be made unique. Rebuild the corpus with `python -m scripts.seed_demo`."
            ) from error


def migrate() -> None:
    """Connect with the configured URL and bring the database up to date, in one transaction."""
    import psycopg

    from src.config import get_settings

    settings = get_settings()
    with psycopg.connect(
        settings.database_url, connect_timeout=settings.db_connect_timeout_seconds
    ) as conn:
        ensure_schema(conn)


def main() -> int:
    migrate()
    print(f"schema is current ({len(STATEMENTS)} idempotent statements applied)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
