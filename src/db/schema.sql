-- Internal Brain — Postgres schema
-- See CLAUDE.md §6 for the tamper-evident audit log design, §1 for the ACL enforcement rule this schema exists to serve.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE roles (
    id              SERIAL PRIMARY KEY,
    name            TEXT UNIQUE NOT NULL,       -- 'support', 'engineering', 'compliance', ...
    clearance_level SMALLINT NOT NULL           -- 0 standard, 1 restricted, ...
);

CREATE TABLE users (
    id       SERIAL PRIMARY KEY,
    name     TEXT NOT NULL,
    email    TEXT UNIQUE NOT NULL,
    role_id  INTEGER REFERENCES roles(id),
    dept     TEXT NOT NULL,
    -- scrypt$<salt>$<hash>, src/api/auth.py. NULL means this account cannot sign in,
    -- which is how a seeded identity exists before anyone gives it a password.
    password_hash TEXT
);

-- Source-of-truth for authorization decisions. Always re-checked at query time — never cached
-- on a document_chunks/transactions row. This is what makes live permission revocation work.
CREATE TABLE permissions (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,
    source_platform TEXT NOT NULL,              -- 'confluence' | 'jira' | 'slack' | 'drive' | 'internal'
    source_ref      TEXT NOT NULL,              -- native id: space/page, project/issue, channel, file/folder
    granted_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at      TIMESTAMPTZ,                -- NULL = currently granted
    -- 'source' when a sync revoked it because the source stopped granting access. NULL for a
    -- live row and for an officer's revoke: a sync never re-grants over a deliberate local
    -- revoke, only restores what it took away itself (src/ingestion/sync.py).
    revoked_by      TEXT
);
CREATE INDEX ON permissions (user_id, source_platform, source_ref);

CREATE TABLE documents (
    id              SERIAL PRIMARY KEY,
    title           TEXT NOT NULL,
    source_platform TEXT NOT NULL,              -- 'confluence' | 'jira' | 'slack' | 'drive' | 'internal'
    source_ref      TEXT NOT NULL,              -- native id, matches permissions.source_ref
    dept            TEXT NOT NULL,
    sensitivity     TEXT NOT NULL,               -- 'internal' | 'restricted'
    acl_tags        TEXT[] NOT NULL,             -- normalized depts/roles allowed to see this doc
    created_at      TIMESTAMPTZ DEFAULT now(),
    -- Freshness (src/ingestion/sync.py). content_hash is SHA-256 of the whitespace-normalised
    -- text mirrored here: how a sync tells "edited" from "unchanged" without re-embedding
    -- everything. NULL on a row written before the column existed; the sync adopts it.
    content_hash      TEXT,
    source_updated_at TIMESTAMPTZ,                -- when the SOURCE says it last changed, if it says
    synced_at         TIMESTAMPTZ DEFAULT now()   -- when this version reached the mirror
);
-- The sync addresses a document by what the source calls it.
CREATE UNIQUE INDEX documents_source_key ON documents (source_platform, source_ref);

CREATE TABLE document_chunks (
    id           SERIAL PRIMARY KEY,
    document_id  INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    content      TEXT NOT NULL,
    embedding    VECTOR(1024) NOT NULL,         -- dimension matches HUNYUAN_EMBEDDING_MODEL
    acl_tags     TEXT[] NOT NULL                -- inherited from documents.acl_tags at ingest time
);
-- hnsw rather than ivfflat: ivfflat builds its cluster lists from the rows present
-- at CREATE INDEX time, and this file runs against an empty database at container
-- init, which would leave the index untrained. hnsw builds incrementally.
CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Structured data source for the SQL Tool agent (fictional Aurelia Financial transactions)
CREATE TABLE transactions (
    id              BIGSERIAL PRIMARY KEY,
    account_dept    TEXT NOT NULL,               -- department the transaction is scoped to
    amount          NUMERIC(14,2) NOT NULL,
    currency        TEXT NOT NULL DEFAULT 'SGD',
    flagged_aml     BOOLEAN NOT NULL DEFAULT false,
    occurred_at     TIMESTAMPTZ NOT NULL,
    acl_tags        TEXT[] NOT NULL              -- departments/roles allowed to query this row
);
CREATE INDEX ON transactions (occurred_at);
CREATE INDEX ON transactions (flagged_aml);

-- Tamper-evident: row_hash = SHA-256 over canonical JSON of [prev_hash, request_id, event_type, user_id,
-- payload, created_at] (src/agents/audit.py), computed at insert time by the Audit agent, never by the database. verify_audit_chain() (application code)
-- walks this table in id order and flags the first row whose row_hash doesn't recompute cleanly.
CREATE TABLE audit_log (
    id          BIGSERIAL PRIMARY KEY,
    request_id  UUID NOT NULL,
    event_type  TEXT NOT NULL,                  -- 'query_received' | 'retrieval' | 'sql_executed' |
                                                 -- 'draft_answer' | 'verification' | 'permission_conflict' |
                                                 -- 'escalation' | 'final_answer'
    user_id     INTEGER REFERENCES users(id),
    payload     JSONB NOT NULL,                 -- includes the detailed `explanation` for escalation/permission_conflict events
    prev_hash   CHAR(64),
    row_hash    CHAR(64) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON audit_log (request_id);

CREATE TABLE escalations (
    id          SERIAL PRIMARY KEY,
    request_id  UUID NOT NULL,
    reason      TEXT NOT NULL,                  -- 'permission_conflict' | 'insufficient_evidence' | 'unsupported' | 'low_confidence'
    status      TEXT NOT NULL DEFAULT 'pending',-- 'pending' | 'reviewed' | 'dismissed'
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- One row per sync attempt per source: what the Sources page calls "last synced", and the
-- evidence that the bounded freshness window is actually being kept.
CREATE TABLE source_syncs (
    id              BIGSERIAL PRIMARY KEY,
    source_platform TEXT NOT NULL,
    trigger         TEXT NOT NULL,               -- 'schedule' | 'manual' | 'seed'
    started_at      TIMESTAMPTZ NOT NULL,
    finished_at     TIMESTAMPTZ NOT NULL,
    added           INTEGER NOT NULL DEFAULT 0,
    updated         INTEGER NOT NULL DEFAULT 0,  -- content re-embedded, or ACL/metadata changed
    removed         INTEGER NOT NULL DEFAULT 0,
    unchanged       INTEGER NOT NULL DEFAULT 0,
    grants_added    INTEGER NOT NULL DEFAULT 0,
    grants_revoked  INTEGER NOT NULL DEFAULT 0,
    error           TEXT                         -- NULL = succeeded
);
CREATE INDEX ON source_syncs (source_platform, finished_at DESC);

-- A single row. `epoch` goes up whenever a sync changes what answers could say, and is part
-- of the answer cache's fingerprint (src/cache.py), so a cached answer from before an edit
-- stops matching in every process at once with no message passed between them.
CREATE TABLE corpus_state (
    id          SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    epoch       BIGINT NOT NULL DEFAULT 0,
    changed_at  TIMESTAMPTZ
);
INSERT INTO corpus_state (id, epoch) VALUES (1, 0);
