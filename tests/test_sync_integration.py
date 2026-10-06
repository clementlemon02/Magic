"""The source sync against real Postgres: the SQL, the transaction and the grants.

    RUN_DATABASE_INTEGRATION=1 .venv/bin/python -m pytest tests/test_sync_integration.py -q

Everything runs inside one transaction that is rolled back, so the database's own documents,
grants and audit chain are left as they were. The source is a fake on the `internal`
platform, which no real connector owns; users are the seeded personas, whose ids 1 and 2 are
load-bearing (scripts/seed_users.sql).
"""

import os
from contextlib import contextmanager

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents import audit
from src.api.auth import build_dependencies
from src.api.compliance import REVOKE_SQL as OFFICER_REVOKE_SQL
from src.api.compliance import build_router
from src.config import get_settings
from src.connectors.base import ConfluencePermission, SourceItem
from src.db.migrate import ensure_schema
from src.graph.state import UserContext
from src.ingestion.sync import SYNC_LOCK, sync_sources
from tests.helpers import as_user

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DATABASE_INTEGRATION") != "1",
    reason="set RUN_DATABASE_INTEGRATION=1 after starting the local Docker database",
)

ALEX, MARCUS = 1, 2  # support, compliance


@pytest.fixture
def connect():
    try:
        conn = psycopg.connect(get_settings().database_url)
    except psycopg.OperationalError as error:
        pytest.skip(f"local database is unavailable: {error}")

    # Open the outer transaction now, or the first conn.transaction() inside the code
    # under test is the OUTER one and commits for real.
    conn.execute("SELECT 1")

    @contextmanager
    def shared():
        # The same connection every time, never closed or committed: the code under test
        # opens conn.transaction() blocks, which become savepoints inside the outer one.
        yield conn

    try:
        yield shared
    finally:
        conn.rollback()
        conn.close()


class FakeSource:
    platform = "internal"

    def __init__(self):
        self.items: dict[str, tuple[str, list[str], str]] = {}

    def put(self, ref, content, groups, sensitivity="internal"):
        self.items[ref] = (content, list(groups), sensitivity)
        return self

    def list_items(self):
        return [
            SourceItem(platform="internal", source_ref=ref, title=f"Doc {ref}", content=content,
                       dept="support", sensitivity=sensitivity)
            for ref, (content, _, sensitivity) in self.items.items()
        ]

    def permissions_for(self, item):
        return ConfluencePermission(space="X", page=item.source_ref,
                                    viewer_groups=self.items[item.source_ref][1])

    def check_access(self, user: UserContext, item):
        return bool(set(user.acl_tags()) & set(self.items[item.source_ref][1]))


class Embeddings:
    """1024 dimensions, as document_chunks.embedding demands; counts what it was asked."""

    def __init__(self):
        self.texts: list[str] = []
        self.fail = False

    def embed_documents(self, texts):
        if self.fail:
            raise ConnectionError("the embedding model is down")
        self.texts += texts
        return [[1.0] + [0.0] * 1023 for _ in texts]


@pytest.fixture
def embeddings():
    return Embeddings()


def run(connect, source, embeddings, **kwargs):
    return sync_sources(connect=connect, embeddings=embeddings,
                        sources={"internal": source}, **kwargs)[0]


def _docs(connect):
    with connect() as conn:
        return {ref: (id_, title, tags) for id_, ref, title, tags in conn.execute(
            "SELECT id, source_ref, title, acl_tags FROM documents "
            "WHERE source_platform = 'internal'").fetchall()}


def _chunks(connect, document_id):
    with connect() as conn:
        return conn.execute(
            "SELECT content, acl_tags FROM document_chunks WHERE document_id = %s ORDER BY id",
            (document_id,)).fetchall()


def _live(connect):
    with connect() as conn:
        return set(conn.execute(
            "SELECT user_id, source_ref FROM permissions "
            "WHERE source_platform = 'internal' AND revoked_at IS NULL").fetchall())


def _epoch(connect):
    with connect() as conn:
        return conn.execute("SELECT epoch FROM corpus_state WHERE id = 1").fetchone()[0]


def _sync_audit_rows(connect):
    with connect() as conn:
        return conn.execute(
            "SELECT user_id, payload FROM audit_log WHERE event_type = 'source_sync' "
            "AND payload->>'platform' = 'internal' ORDER BY id").fetchall()


def _two():
    return FakeSource().put("a", "alpha beta gamma", ["support"]).put("b", "delta epsilon", ["compliance"])


# --- loading ---------------------------------------------------------------------------

def test_the_first_sync_loads_documents_chunks_and_grants(connect, embeddings):
    report = run(connect, _two(), embeddings)

    assert (report.added, report.updated, report.removed, report.error) == (2, 0, 0, None)
    docs = _docs(connect)
    assert sorted(docs) == ["a", "b"]
    assert docs["a"][2] == ["support"]
    assert [c[0] for c in _chunks(connect, docs["a"][0])] == ["alpha beta gamma"]
    live = _live(connect)
    assert (ALEX, "a") in live and (MARCUS, "b") in live
    assert (ALEX, "b") not in live and (MARCUS, "a") not in live


def test_the_second_sync_finds_nothing_to_do(connect, embeddings):
    """A scheduled run that re-embedded the corpus, or bumped the epoch, every ten minutes
    would empty the answer cache on a timer."""
    source = _two()
    run(connect, source, embeddings)
    epoch, embedded, audited = _epoch(connect), len(embeddings.texts), len(_sync_audit_rows(connect))

    report = run(connect, source, embeddings, trigger="schedule")

    assert (report.added, report.updated, report.removed, report.unchanged) == (0, 0, 0, 2)
    assert (report.grants_added, report.grants_revoked) == (0, 0)
    assert _epoch(connect) == epoch
    assert len(embeddings.texts) == embedded, "unchanged text was embedded again"
    assert len(_sync_audit_rows(connect)) == audited, "a quiet scheduled run wrote an audit row"


# --- edits -----------------------------------------------------------------------------

def test_an_edit_replaces_the_text_in_place_and_bumps_the_epoch(connect, embeddings):
    source = _two()
    run(connect, source, embeddings)
    before = _docs(connect)["a"][0]
    epoch = _epoch(connect)

    source.put("a", "alpha beta CHANGED", ["support"])
    report = run(connect, source, embeddings, trigger="schedule")

    assert (report.added, report.updated, report.removed, report.unchanged) == (0, 1, 0, 1)
    docs = _docs(connect)
    assert docs["a"][0] == before, "the document must keep its id, or old citations stop resolving"
    assert [c[0] for c in _chunks(connect, before)] == ["alpha beta CHANGED"]
    assert _epoch(connect) == epoch + 1
    assert embeddings.texts[-1] == "alpha beta CHANGED"

    user_id, payload = _sync_audit_rows(connect)[-1]
    assert payload["rewritten"] == ["a"] and payload["added"] == payload["removed"] == []
    assert user_id is None, "a scheduled sync has no officer to attribute it to"


def test_a_long_document_is_rechunked(connect, embeddings):
    source = FakeSource().put("long", " ".join(f"w{i}" for i in range(450)), ["support"])
    run(connect, source, embeddings)
    assert len(_chunks(connect, _docs(connect)["long"][0])) == 3  # 200 + 200 + 50 words

    source.put("long", "now short", ["support"])
    run(connect, source, embeddings)
    assert [c[0] for c in _chunks(connect, _docs(connect)["long"][0])] == ["now short"]


# --- who may read ----------------------------------------------------------------------

def test_narrowing_access_retags_chunks_and_revokes_grants_without_re_embedding(connect, embeddings):
    source = _two()
    run(connect, source, embeddings)
    embedded = len(embeddings.texts)
    epoch = _epoch(connect)
    assert (ALEX, "a") in _live(connect)

    source.put("a", "alpha beta gamma", ["compliance"])  # same words, narrower audience
    report = run(connect, source, embeddings, trigger="schedule")

    assert report.updated == 1 and report.grants_revoked >= 1
    doc_id, _, tags = _docs(connect)["a"]
    assert tags == ["compliance"]
    assert [c[1] for c in _chunks(connect, doc_id)] == [["compliance"]], "chunks must carry the new ACL"
    assert (ALEX, "a") not in _live(connect) and (MARCUS, "a") in _live(connect)
    assert len(embeddings.texts) == embedded, "an ACL change does not change the words"
    assert _epoch(connect) == epoch + 1

    with connect() as conn:
        revoked_by = conn.execute(
            "SELECT revoked_by FROM permissions WHERE user_id = %s AND source_ref = 'a' "
            "AND revoked_at IS NOT NULL", (ALEX,)).fetchone()[0]
    assert revoked_by == "source"
    payload = _sync_audit_rows(connect)[-1][1]
    assert {"user_id": ALEX, "source_ref": "a"} in payload["revoked"]


def test_access_a_sync_revoked_returns_when_the_source_restores_it(connect, embeddings):
    source = _two()
    run(connect, source, embeddings)
    source.put("a", "alpha beta gamma", ["compliance"])
    run(connect, source, embeddings)
    assert (ALEX, "a") not in _live(connect)

    source.put("a", "alpha beta gamma", ["support"])
    run(connect, source, embeddings)
    assert (ALEX, "a") in _live(connect)


def test_a_revoke_an_officer_made_survives_a_sync(connect, embeddings):
    source = _two()
    run(connect, source, embeddings)
    with connect() as conn:
        changed = conn.execute(OFFICER_REVOKE_SQL, {
            "user_id": ALEX, "source_platform": "internal", "source_ref": "a"}).rowcount
    assert changed == 1

    report = run(connect, source, embeddings, trigger="schedule")  # the source still backs it

    assert (ALEX, "a") not in _live(connect)
    assert report.grants_added == 0, "a sync re-granted what an officer had revoked"


# --- removal ---------------------------------------------------------------------------

def test_a_deleted_item_leaves_no_chunks_and_no_live_grants(connect, embeddings):
    source = _two()
    run(connect, source, embeddings)
    gone = _docs(connect)["a"][0]

    del source.items["a"]
    report = run(connect, source, embeddings, trigger="schedule")

    assert report.removed == 1
    assert "a" not in _docs(connect)
    assert _chunks(connect, gone) == []
    assert all(ref != "a" for _, ref in _live(connect))
    assert "b" in _docs(connect), "removing one item must not touch another"


# --- failure ---------------------------------------------------------------------------

def test_a_source_that_cannot_be_listed_is_not_read_as_empty(connect, embeddings):
    """The dangerous misreading: a down source returns nothing, and "nothing" is
    indistinguishable from "everything was deleted". The platform rolls back untouched."""
    source = _two()
    run(connect, source, embeddings)

    class Down(FakeSource):
        def list_items(self):
            raise ConnectionError("the source is down")

    report = run(connect, Down(), embeddings, trigger="schedule")

    assert report.error and "the source is down" in report.error
    assert sorted(_docs(connect)) == ["a", "b"]
    assert (ALEX, "a") in _live(connect)
    with connect() as conn:
        errors = conn.execute("SELECT error FROM source_syncs WHERE source_platform = 'internal' "
                              "ORDER BY id").fetchall()
    assert errors[-1][0] and "the source is down" in errors[-1][0]


def test_an_embedding_failure_leaves_the_mirror_as_it_was(connect, embeddings):
    source = _two()
    run(connect, source, embeddings)
    epoch = _epoch(connect)

    source.put("a", "alpha beta EDITED", ["support"])
    embeddings.fail = True
    report = run(connect, source, embeddings, trigger="schedule")

    assert report.error
    assert [c[0] for c in _chunks(connect, _docs(connect)["a"][0])] == ["alpha beta gamma"]
    assert _epoch(connect) == epoch, "a rolled-back sync must not invalidate the cache"


def test_one_failing_source_does_not_stop_the_others(connect, embeddings):
    class Down(FakeSource):
        def list_items(self):
            raise ConnectionError("the source is down")

    reports = sync_sources(connect=connect, embeddings=embeddings,
                           sources={"first": Down(), "second": _two()}, trigger="schedule")

    assert reports[0].error and reports[1].error is None
    assert sorted(_docs(connect)) == ["a", "b"]


def test_a_source_already_being_synced_is_skipped(connect, embeddings):
    """Two syncs of one source — the schedule and a click, or two workers — must not race."""
    other = psycopg.connect(get_settings().database_url, autocommit=True)
    other.execute("SELECT pg_advisory_lock(%s, hashtext('internal'))", (SYNC_LOCK,))
    try:
        report = run(connect, _two(), embeddings, trigger="schedule")
    finally:
        other.execute("SELECT pg_advisory_unlock(%s, hashtext('internal'))", (SYNC_LOCK,))
        other.close()

    assert report.skipped and report.error is None
    assert _docs(connect) == {}


# --- the record ------------------------------------------------------------------------

def test_a_manual_sync_is_on_the_record_even_when_nothing_moved(connect, embeddings):
    source = _two()
    run(connect, source, embeddings)
    before = len(_sync_audit_rows(connect))

    run(connect, source, embeddings, trigger="manual", user_id=MARCUS)

    rows = _sync_audit_rows(connect)
    assert len(rows) == before + 1
    assert rows[-1][0] == MARCUS and rows[-1][1]["trigger"] == "manual"


def test_the_audit_chain_still_verifies_after_syncs(connect, embeddings):
    source = _two()
    run(connect, source, embeddings)
    source.put("a", "alpha EDITED", ["compliance"])
    run(connect, source, embeddings, trigger="manual", user_id=MARCUS)

    report = audit.verify_audit_chain(connect)
    assert report.ok, report.problem


def test_the_migration_can_be_applied_twice(connect):
    with connect() as conn:
        ensure_schema(conn)
        ensure_schema(conn)


# --- what the Sources page reads -------------------------------------------------------

def test_the_sources_screen_reports_the_last_sync_and_a_failure_since(connect, embeddings):
    source = _two()
    run(connect, source, embeddings, trigger="schedule")

    marcus = UserContext(id=MARCUS, role="compliance", dept="compliance", clearance_level=1)
    app = FastAPI()
    _, officer = build_dependencies(lambda uid: marcus)
    app.include_router(build_router(lambda uid: marcus, connect=connect, officer=officer))
    client = TestClient(app)

    def internal():
        rows = client.get("/admin/sources", headers=as_user(MARCUS)).json()
        return next(r for r in rows if r["platform"] == "internal")

    ok = internal()
    assert ok["documents"] == 2 and ok["last_synced"] and ok["stale"] is False
    assert ok["last_error"] is None

    class Down(FakeSource):
        def list_items(self):
            raise ConnectionError("the source is down")

    run(connect, Down(), embeddings, trigger="schedule")
    failing = internal()
    assert "the source is down" in failing["last_error"]
    assert failing["last_synced"] == ok["last_synced"], "a failure must not move 'last synced'"
