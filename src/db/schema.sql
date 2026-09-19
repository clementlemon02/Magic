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
    dept     TEXT NOT NULL
);

-- Source-of-truth for authorization decisions. Always re-checked at query time — never cached
-- on a document_chunks/transactions row. This is what makes live permission revocation work.
CREATE TABLE permissions (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,
    source_platform TEXT NOT NULL,              -- 'confluence' | 'jira' | 'slack' | 'drive' | 'internal'
    source_ref      TEXT NOT NULL,              -- native id: space/page, project/issue, channel, file/folder
    granted_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at      TIMESTAMPTZ                 -- NULL = currently granted
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
    created_at      TIMESTAMPTZ DEFAULT now()
);

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

-- Tamper-evident: row_hash = SHA-256(prev_hash || request_id || event_type || payload || created_at),
-- computed at insert time by the Audit agent, never by the database. verify_audit_chain() (application code)
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
    reason      TEXT NOT NULL,                  -- 'permission_conflict' | 'low_confidence' | 'sensitivity_exceeded'
    status      TEXT NOT NULL DEFAULT 'pending',-- 'pending' | 'reviewed' | 'dismissed'
    created_at  TIMESTAMPTZ DEFAULT now()
);
