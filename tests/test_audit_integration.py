"""The audit chain against real Postgres: JSONB round-trip, tamper detection, revoke.

    RUN_DATABASE_INTEGRATION=1 .venv/bin/python -m pytest tests/test_audit_integration.py -q

Everything runs inside one transaction that is rolled back, so the dev database's
own chain is left as it was.
"""

import os
import uuid
from contextlib import contextmanager

import psycopg
import pytest

from src.agents import audit
from src.agents.escalation import escalation_node
from src.config import get_settings
from src.graph.state import PermConflict, UserContext

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DATABASE_INTEGRATION") != "1",
    reason="set RUN_DATABASE_INTEGRATION=1 after starting the local Docker database",
)

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)


@pytest.fixture
def connect():
    try:
        conn = psycopg.connect(get_settings().database_url)
    except psycopg.OperationalError as error:
        pytest.skip(f"local database is unavailable: {error}")

    # Open the outer transaction now. Without this, the first conn.transaction()
    # inside the code under test is the OUTER one and commits for real.
    conn.execute("SELECT 1")

    @contextmanager
    def shared():
        # Same connection every time, never closed or committed: inner
        # conn.transaction() blocks become savepoints inside the outer one.
        yield conn

    try:
        yield shared
    finally:
        conn.rollback()
        conn.close()


def _escalated_state():
    state = {
        "request_id": str(uuid.uuid4()),
        "query": "What triggers an AML escalation review?",
        "user": ALEX,
        "route": "rag",
        "hop_count": 1,
        "retrieved_chunks": [],
        "permission_conflicts": [PermConflict(
            document_id=14, source_platform="confluence",
            source_ref="COMPLIANCE/aml-escalation", sensitivity="restricted",
            score_margin=0.3141592653589793,
        )],
        "sql_result": None, "draft_answer": None, "verification": None,
        "clarification_question": None, "final_answer": None, "citations": [],
        "explanation": None, "escalated": False, "audit_events": [],
    }
    state.update(escalation_node(state))
    return state


def test_a_refusal_writes_a_chain_that_verifies_after_the_jsonb_round_trip(connect):
    state = _escalated_state()
    audit.audit_node(state, connect=connect)

    assert audit.verify_audit_chain(connect).ok
    events = audit.read_request(state["request_id"], connect)
    assert "COMPLIANCE/aml-escalation" in next(
        e.payload["explanation"] for e in events if e.event_type == "escalation"
    )
    with connect() as conn:
        reason = conn.execute(
            "SELECT reason FROM escalations WHERE request_id = %s", (state["request_id"],)
        ).fetchone()
    assert reason == ("permission_conflict",)


def test_editing_a_row_in_the_database_is_caught(connect):
    state = _escalated_state()
    audit.audit_node(state, connect=connect)
    with connect() as conn:
        target = conn.execute(
            "SELECT id FROM audit_log WHERE request_id = %s AND event_type = 'escalation'",
            (state["request_id"],),
        ).fetchone()[0]
        # What a DBA covering tracks would do: rewrite the reason in place.
        conn.execute(
            "UPDATE audit_log SET payload = jsonb_set(payload, '{reason}', '\"low_confidence\"') "
            "WHERE id = %s",
            (target,),
        )

    report = audit.verify_audit_chain(connect)
    assert not report.ok and report.first_bad_id == target
