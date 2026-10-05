"""The audit chain against real Postgres: JSONB round-trip, tamper detection, revoke.

    RUN_DATABASE_INTEGRATION=1 .venv/bin/python -m pytest tests/test_audit_integration.py -q

Everything runs inside one transaction that is rolled back, so the dev database's
own chain is left as it was.
"""

import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from src.agents import audit
from src.agents.escalation import escalation_node
from src.config import get_settings
from src.graph.state import AuditEvent, PermConflict, UserContext

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


# --- the officer's inquiry: who, and which documents --------------------------------
#
# Real SQL, because the query is a CTE over four differently shaped JSONB payloads and a
# fake connection cannot catch a mistake in that. Refs carry a per-test tag: the dev
# database holds real activity, and the worklist reads every row in its window.

def _tag() -> str:
    return f"af-{uuid.uuid4().hex[:8]}"


def _request(connect, user_id, query, *, cites=(), retrieved=(), conflicts=(), rechecked=(),
             text="an answer", reason=None):
    """Write one request's trail the way the graph would, and return its id."""
    request_id = str(uuid.uuid4())
    at = datetime.now(UTC)

    def event(kind, payload):
        return AuditEvent(event_type=kind, payload=payload, occurred_at=at)

    def named(refs):
        return [{"source_platform": p, "source_ref": r} for p, r in (x.split(":", 1) for x in refs)]

    events = [event("query_received", {"query": query, "route": "rag"})]
    if retrieved:
        events.append(event("retrieval", {"hops": 1, "chunk_ids": [], "document_ids": [],
                                          "sources": list(retrieved), "scores": []}))
    if conflicts:
        events.append(event("permission_conflict", {"conflicts": [
            {**item, "sensitivity": "restricted", "document_id": 1, "score_margin": 0.2}
            for item in named(conflicts)]}))
    if rechecked:
        events.append(event("source_recheck_denied", {"items": named(rechecked)}))
    if reason:
        events.append(event("escalation", {"reason": reason}))
    events.append(event("final_answer", {"escalated": bool(reason), "text": text,
                                         "citations": named(cites)}))
    audit.append_events(request_id, user_id, events, escalation_reason=reason, connect=connect)
    return request_id


def _worklist(connect, **filters):
    since = datetime.now(UTC) - timedelta(days=1)
    return audit.recent_requests(since, connect=connect, **filters)


def _ids(connect, **filters):
    return {row["request_id"] for row in _worklist(connect, **filters)}


def test_a_person_is_found_by_id_name_or_email_and_named_on_the_row(connect):
    tag = _tag()
    alex = _request(connect, 1, f"alex asks {tag}", cites=[f"confluence:TEST/{tag}"])
    marcus = _request(connect, 2, f"marcus asks {tag}", cites=[f"confluence:TEST/{tag}"])

    assert _ids(connect, document=tag) == {alex, marcus}
    assert _ids(connect, document=tag, user="1") == {alex}
    assert _ids(connect, document=tag, user="alex") == {alex}
    assert _ids(connect, document=tag, user="marcus.lim@") == {marcus}
    assert _ids(connect, document=tag, user="nobody by that name") == set()

    [row] = _worklist(connect, document=tag, user="1")
    assert row["user_id"] == 1 and row["user_name"] == "Alex Tan"


def test_a_document_is_found_wherever_a_request_names_it(connect):
    """Cited, retrieved, refused because it outranked the permitted evidence, and denied by
    the source at question time: four different JSONB shapes, one filter."""
    tag = _tag()
    cited = _request(connect, 1, "cited", cites=[f"confluence:TEST/{tag}-cited"])
    retrieved = _request(connect, 1, "retrieved", retrieved=[f"jira:TEST/{tag}-retrieved"])
    refused = _request(connect, 1, "refused", conflicts=[f"confluence:TEST/{tag}-refused"],
                       reason="permission_conflict", text="I can't answer that.")
    rechecked = _request(connect, 1, "rechecked", rechecked=[f"drive:TEST/{tag}-rechecked"])
    # Mentions a document in prose and touched nothing: must never match.
    _request(connect, 1, "mentions", text=f"see confluence:TEST/{tag}-cited for details")

    assert _ids(connect, document=f"{tag}-cited") == {cited}
    assert _ids(connect, document=f"jira:TEST/{tag}-retrieved") == {retrieved}
    assert _ids(connect, document=f"{tag}-refused") == {refused}
    assert _ids(connect, document=f"drive:TEST/{tag}-rechecked") == {rechecked}
    # A whole "space" is a prefix, and platform narrows it.
    assert _ids(connect, document=f"TEST/{tag}") == {cited, retrieved, refused, rechecked}
    assert _ids(connect, document=f"confluence:TEST/{tag}") == {cited, refused}
    # The inquiry composes with the outcome filter: what was turned away at that space.
    assert _ids(connect, document=f"confluence:TEST/{tag}", outcome="refused") == {refused}
    assert _ids(connect, document=f"confluence:TEST/{tag}", outcome="answered") == {cited}


def test_who_and_which_document_compose_into_the_briefs_inquiry(connect):
    """The brief's inquiry: everything user jdoe accessed related to the payment-gateway space."""
    tag = _tag()
    mine = _request(connect, 1, "mine", cites=[f"confluence:TEST/{tag}-a"])
    _request(connect, 2, "someone else, same space", cites=[f"confluence:TEST/{tag}-b"])
    _request(connect, 1, "mine, other space", cites=[f"confluence:ELSEWHERE/{tag}"])

    assert _ids(connect, user="1", document=f"confluence:TEST/{tag}") == {mine}


def test_the_filters_apply_before_the_limit(connect):
    """Three of someone's requests, then five newer ones from somebody else. A filter applied
    to the newest three rows would find none of them."""
    tag = _tag()
    for i in range(3):
        _request(connect, 2, f"marcus {i}", cites=[f"confluence:TEST/{tag}"])
    for i in range(5):
        _request(connect, 1, f"alex {i}", cites=[f"confluence:TEST/{tag}"])

    rows = _worklist(connect, document=tag, user="marcus", limit=3)
    assert len(rows) == 3 and {r["user_id"] for r in rows} == {2}


def test_a_wildcard_in_a_document_filter_is_a_literal(connect):
    tag = _tag()
    _request(connect, 1, "q", cites=[f"confluence:TEST/{tag}-x"])
    assert _ids(connect, document=f"{tag}-%") == set()
    assert _ids(connect, document=f"{tag}_x") == set()
    assert len(_ids(connect, document=f"{tag}-x")) == 1


def test_all_digits_is_an_id_and_nothing_else(connect):
    """"8" must not also find the person whose name or address merely contains an 8."""
    tag = _tag()
    with connect() as conn:
        eight = conn.execute(
            "INSERT INTO users (name, email, role_id, dept) "
            "VALUES (%s, %s, (SELECT id FROM roles LIMIT 1), 'support') RETURNING id",
            (f"Eight 8 {tag}", f"eight8-{tag}@example.test"),
        ).fetchone()[0]
    assert eight != 8
    mine = _request(connect, eight, "digits in the name", cites=[f"confluence:TEST/{tag}"])

    assert _ids(connect, document=tag, user="8") == set()
    assert _ids(connect, document=tag, user=str(eight)) == {mine}
    assert _ids(connect, document=tag, user="Eight") == {mine}
    assert _ids(connect, document=tag, user=f"eight8-{tag}") == {mine}
