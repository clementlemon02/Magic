"""The access-gap report: which documents people keep being refused.

    RUN_DATABASE_INTEGRATION=1 .venv/bin/python -m pytest tests/test_access_gap.py -q

The aggregation is plain Python, but the query behind it is not — a LATERAL unnest
of two different JSONB array shapes, unioned. A fake connection cannot catch a
mistake in that, so the ranking is tested in isolation and the SQL for real.
"""

import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from src.agents import audit
from src.agents.access_gap import run_access_gap_scan
from src.config import get_settings
from src.graph.state import AuditEvent, UserContext

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)
MARCUS = UserContext(id=2, role="compliance", dept="compliance", clearance_level=1)
AML = {"source_platform": "confluence", "source_ref": "COMPLIANCE/aml-escalation"}
LEDGER = {"source_platform": "drive", "source_ref": "file-ledger"}


# --- ranking, with no database ------------------------------------------------

def _fake_connect(rows):
    class Conn:
        def execute(self, sql, params):
            return rows

    @contextmanager
    def connect():
        yield Conn()

    return connect


def _row(ref, request_id, user_id, query, platform="confluence", signal="permission_conflict"):
    return (platform, ref, "restricted", signal, request_id, user_id, query)


def test_ranked_by_distinct_askers_not_by_request_count():
    """One person retrying is a bad afternoon; four people is a broken permission model."""
    rows = [
        # One asker, hammering the same question four times.
        *[_row("noisy", f"r{i}", 1, "again?") for i in range(4)],
        # Three different people, once each.
        *[_row("shared", f"s{i}", 10 + i, "what is the AML threshold?") for i in range(3)],
    ]
    report = run_access_gap_scan(datetime.now(UTC), connect=_fake_connect(rows))
    assert [g.source_ref for g in report.gaps] == ["shared", "noisy"]
    assert report.gaps[0].askers == 3 and report.gaps[0].requests == 3
    assert report.gaps[1].askers == 1 and report.gaps[1].requests == 4


def test_both_signals_on_one_document_are_reported_together():
    rows = [
        _row("drifted", "r1", 1, "q1"),
        _row("drifted", "r2", 2, "q2", signal="source_recheck_denied"),
    ]
    gap = run_access_gap_scan(datetime.now(UTC), connect=_fake_connect(rows)).gaps[0]
    assert gap.signals == ["permission_conflict", "source_recheck_denied"]
    assert gap.requests == 2 and gap.askers == 2


def test_example_queries_are_deduplicated_and_capped():
    rows = [_row("x", f"r{i}", i, "the same question") for i in range(5)]
    rows.append(_row("x", "r9", 9, "a different one"))
    gap = run_access_gap_scan(datetime.now(UTC), connect=_fake_connect(rows)).gaps[0]
    assert gap.example_queries == ["the same question", "a different one"]


def test_no_refusals_is_an_empty_report():
    report = run_access_gap_scan(datetime.now(UTC), connect=_fake_connect([]))
    assert report.gaps == [] and report.refusals_scanned == 0


# --- the real query -----------------------------------------------------------

# On the two tests below, not the module: the ranking above needs no database.
needs_database = pytest.mark.skipif(
    os.getenv("RUN_DATABASE_INTEGRATION") != "1",
    reason="set RUN_DATABASE_INTEGRATION=1 after starting the local Docker database",
)


@pytest.fixture
def connect():
    try:
        conn = psycopg.connect(get_settings().database_url)
    except psycopg.OperationalError as error:
        pytest.skip(f"local database is unavailable: {error}")
    conn.execute("SELECT 1")

    @contextmanager
    def shared():
        yield conn

    try:
        yield shared
    finally:
        conn.rollback()
        conn.close()


def _refusal(connect, user: UserContext, query: str, event: AuditEvent) -> None:
    """One request's worth of audit rows: the question, then why it was refused."""
    now = datetime.now(UTC)
    with connect() as conn, conn.transaction():
        audit.append_in(conn, str(uuid.uuid4()), user.id, [
            AuditEvent(event_type="query_received", payload={"query": query}, occurred_at=now),
            event,
        ])


def _conflict(**item) -> AuditEvent:
    return AuditEvent(
        event_type="permission_conflict",
        payload={"conflicts": [{**item, "document_id": 1, "score_margin": 0.3}]},
        occurred_at=datetime.now(UTC),
    )


def _recheck_denied(**item) -> AuditEvent:
    return AuditEvent(
        event_type="source_recheck_denied",
        payload={"items": [item]},
        occurred_at=datetime.now(UTC),
    )


@needs_database
def test_the_query_unnests_both_json_shapes_and_ranks_them(connect):
    _refusal(connect, ALEX, "what triggers an AML review?",
             _conflict(**AML, sensitivity="restricted"))
    _refusal(connect, MARCUS, "AML threshold?", _conflict(**AML, sensitivity="restricted"))
    _refusal(connect, ALEX, "who is on the ledger?", _recheck_denied(**LEDGER))

    report = run_access_gap_scan(datetime.now(UTC) - timedelta(hours=1), connect=connect)
    by_ref = {g.source_ref: g for g in report.gaps}

    assert report.refusals_scanned == 3
    # The AML page: two different people, so it ranks above the one-asker ledger.
    assert [g.source_ref for g in report.gaps] == [AML["source_ref"], LEDGER["source_ref"]]
    aml = by_ref[AML["source_ref"]]
    assert aml.askers == 2 and aml.requests == 2
    assert aml.sensitivity == "restricted"
    assert aml.signals == ["permission_conflict"]
    assert "what triggers an AML review?" in aml.example_queries
    # The source-recheck signal survives its different payload shape.
    assert by_ref[LEDGER["source_ref"]].signals == ["source_recheck_denied"]


@needs_database
def test_refusals_before_the_window_are_not_counted(connect):
    _refusal(connect, ALEX, "old question", _conflict(**AML, sensitivity="restricted"))
    report = run_access_gap_scan(datetime.now(UTC) + timedelta(hours=1), connect=connect)
    assert report.gaps == []
