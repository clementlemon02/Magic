"""Escalation, the audit hash chain, and the compliance route guards. No database."""

from datetime import UTC, datetime, timedelta
from functools import partial

import pytest
from fastapi.testclient import TestClient

from src.agents.audit import events_for, row_hash, verify_rows
from src.agents.escalation import escalation_node, escalation_reason
from src.agents.knowledge_gap import cluster
from src.agents.router import route_node
from src.agents.synthesizer import synthesize_node
from src.agents.verifier import verify_node
from src.api.main import create_app
from src.config import get_settings
from src.graph.state import (
    GENERIC_REFUSAL,
    Chunk,
    Citation,
    PermConflict,
    UserContext,
    VerificationResult,
)
from src.llm.fake import FakeChatModel
from tests.helpers import as_user

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)
MARCUS = UserContext(id=2, role="compliance", dept="compliance", clearance_level=1)
CONFLICT = PermConflict(
    document_id=7,
    source_platform="confluence",
    source_ref="COMPLIANCE/aml-escalation",
    sensitivity="restricted",
    score_margin=0.31,
)


@pytest.fixture(autouse=True)
def pinned_settings(monkeypatch):
    monkeypatch.setenv("VERIFIER_CONFIDENCE_THRESHOLD", "0.6")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _state(**over):
    return {
        "request_id": "00000000-0000-0000-0000-000000000001",
        "query": "q",
        "user": ALEX,
        "route": "rag",
        "hop_count": 1,
        "retrieved_chunks": [],
        "permission_conflicts": [],
        "sql_result": None,
        "draft_answer": None,
        "verification": None,
        "clarification_question": None,
        "final_answer": None,
        "citations": [],
        "explanation": None,
        "escalated": False,
        "audit_events": [],
        **over,
    }


# --- Escalation ---------------------------------------------------------------

@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"permission_conflicts": [CONFLICT]}, "permission_conflict"),
        ({}, "insufficient_evidence"),
        ({"draft_answer": "x", "verification": VerificationResult(grounded=True, confidence=0.3)},
         "low_confidence"),
        ({"draft_answer": "x", "hop_count": 3,
          "verification": VerificationResult(grounded=False, unsupported=["c"], confidence=0.9)},
         "unsupported"),
    ],
)
def test_every_cause_refuses_with_the_same_text_and_its_own_reason(over, reason):
    assert escalation_reason(_state(**over)) == reason
    out = escalation_node(_state(**over))
    assert out["escalated"] is True
    assert out["final_answer"] == GENERIC_REFUSAL
    assert out["citations"] == []
    assert out["audit_events"][-1].payload["reason"] == reason


def test_the_explanation_names_the_withheld_item_for_the_auditor():
    out = escalation_node(_state(permission_conflicts=[CONFLICT]))
    assert "COMPLIANCE/aml-escalation" in out["explanation"]
    assert "clearance 0" in out["explanation"]


def test_the_explanation_never_reaches_the_asker_through_the_real_node():
    """Real Escalation node in the real graph; only the chat model and retrieval stood in."""
    chat = FakeChatModel(route="rag")
    nodes = {
        "router": partial(route_node, chat_model=chat),
        "clarification": lambda s: {},
        "synthesizer": partial(synthesize_node, chat_model=chat),
        "verifier": partial(verify_node, chat_model=chat),
        "sql_tool": lambda s: {},
        "retrieval": lambda s: {"hop_count": 1, "permission_conflicts": [CONFLICT]},
        "escalation": escalation_node,
        "audit": lambda s: {"audit_events": []},
    }
    r = TestClient(create_app(nodes=nodes, user_loader=lambda uid: ALEX)).post(
        "/query", json={"query": "What triggers an AML escalation review?"}, headers=as_user(1)
    )
    # Exact, not a subset: a field added to AskerResponse has to be considered here
    # before it can ride out on a refusal. `trace` is empty on every one of them.
    assert r.json() == {"text": GENERIC_REFUSAL, "citations": [], "trace": []}


# --- The trail ----------------------------------------------------------------

def test_the_trail_records_identity_but_never_chunk_content():
    chunk = Chunk(
        id=10, document_id=1, content="SECRET BODY TEXT", acl_tags=["support"], score=0.8,
        citation=Citation(document_id=1, title="t", source_platform="confluence", source_ref="r"),
    )
    trail = events_for(_state(retrieved_chunks=[chunk], permission_conflicts=[CONFLICT]))
    types = [e.event_type for e in trail]
    assert types[0] == "query_received" and types[-1] == "final_answer"
    assert "permission_conflict" in types
    assert "SECRET BODY TEXT" not in str([e.payload for e in trail])


def test_the_retrieval_event_names_its_documents_not_just_their_ids():
    """A document id means a row in the corpus as it was: reseeding restarts the sequence and
    a sync can remove and re-add a document under a new one. Names are what an officer can
    still ask about a month later. In chunk order, one entry per document, never content."""
    def chunk(id_, doc, ref, platform="confluence"):
        return Chunk(
            id=id_, document_id=doc, content="BODY", acl_tags=["support"], score=0.8,
            citation=Citation(document_id=doc, title="t", source_platform=platform, source_ref=ref),
        )

    trail = events_for(_state(retrieved_chunks=[
        chunk(1, 5, "SUPPORT/refund-policy"), chunk(2, 5, "SUPPORT/refund-policy"),
        chunk(3, 9, "file-pii-standard", platform="drive"),
    ]))
    retrieval = next(e for e in trail if e.event_type == "retrieval").payload
    assert retrieval["sources"] == ["confluence:SUPPORT/refund-policy", "drive:file-pii-standard"]
    assert retrieval["document_ids"] == [5, 9]
    assert "BODY" not in str(retrieval)


def test_escalation_reason_and_explanation_are_carried_into_the_trail():
    escalated = _state(permission_conflicts=[CONFLICT])
    escalated.update(escalation_node(escalated))
    trail = events_for(escalated)
    escalation = next(e for e in trail if e.event_type == "escalation")
    assert escalation.payload["reason"] == "permission_conflict"
    assert "COMPLIANCE/aml-escalation" in escalation.payload["explanation"]


# --- Hash chain ---------------------------------------------------------------

def _chain(n=4):
    rows, prev, t0 = [], None, datetime(2026, 10, 1, tzinfo=UTC)
    for i in range(n):
        args = ("00000000-0000-0000-0000-000000000001", "retrieval", 1, {"i": i, "f": 0.31},
                t0 + timedelta(seconds=i))
        digest = row_hash(prev, *args)
        rows.append([i + 1, args[0], args[1], args[2], args[3], prev, digest, args[4]])
        prev = digest
    return rows


def test_an_untouched_chain_verifies():
    report = verify_rows(_chain())
    assert report.ok and report.rows_checked == 4


@pytest.mark.parametrize(
    ("column", "value"), [(4, {"i": 99, "f": 0.31}), (3, 2), (2, "final_answer")]
)
def test_editing_payload_user_or_type_is_flagged_at_that_row(column, value):
    rows = _chain()
    rows[2][column] = value
    report = verify_rows(rows)
    assert not report.ok and report.first_bad_id == 3


def test_deleting_a_row_is_flagged_at_the_next_one():
    rows = _chain()
    del rows[1]
    report = verify_rows(rows)
    assert not report.ok and report.first_bad_id == 3
    assert "removed" in report.problem


def test_the_hash_does_not_depend_on_the_session_timezone():
    t = datetime(2026, 10, 1, 12, tzinfo=UTC)
    sgt = t.astimezone(__import__("zoneinfo").ZoneInfo("Asia/Singapore"))
    assert row_hash(None, "r", "e", 1, {}, t) == row_hash(None, "r", "e", 1, {}, sgt)


# --- Knowledge gap clustering -------------------------------------------------

def test_similar_questions_cluster_and_distinct_ones_do_not():
    vectors = [[1, 0], [0.98, 0.2], [0, 1], [0.1, 0.99]]
    assert cluster(vectors, threshold=0.9) == [[0, 1], [2, 3]]


# --- Compliance route guards --------------------------------------------------

def _client(user):
    def unused_node(state):
        raise AssertionError("graph must not run")

    nodes = dict.fromkeys(
        ["router", "retrieval", "sql_tool", "clarification", "synthesizer", "verifier",
         "escalation", "audit"], unused_node,
    )
    return TestClient(create_app(nodes=nodes, user_loader=lambda uid: user))


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/audit/00000000-0000-0000-0000-000000000001", None),
        ("get", "/audit/verify", None),
        ("get", "/knowledge-gaps", None),
        ("post", "/admin/permissions/revoke",
         {"user_id": 1, "source_platform": "confluence", "source_ref": "x"}),
        ("post", "/admin/permissions/grant",
         {"user_id": 1, "source_platform": "confluence", "source_ref": "x"}),
    ],
)
# Two different refusals now, and the difference is the point: a signed-in person who
# is not an officer is told so (403), while a token naming nobody is not told whether
# that account exists (401).
@pytest.mark.parametrize(
    ("caller", "expected"),
    [(ALEX, 403), (None, 401)],
    ids=["not-an-officer", "unknown-user"],
)
def test_only_a_compliance_officer_reaches_compliance_routes(method, path, body, caller, expected):
    client, headers = _client(caller), as_user(1)
    r = client.post(path, headers=headers, json=body) if body else client.get(path, headers=headers)
    assert r.status_code == expected
    assert r.status_code != 200


def test_a_malformed_request_id_is_rejected_before_the_database():
    r = _client(MARCUS).get("/audit/not-a-uuid", headers=as_user(2))
    assert r.status_code == 422


# --- narrowing the worklist -----------------------------------------------------

class _FakeConn:
    """Captures the parameters a query was bound with."""

    def __init__(self, store):
        self.store = store

    def execute(self, sql, params):
        self.store["sql"], self.store["params"] = sql, params
        return self

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ran(**kwargs):
    from src.agents.audit import recent_requests

    seen: dict = {}
    recent_requests(datetime.now(UTC), connect=lambda: _FakeConn(seen), **kwargs)
    return seen


def test_the_worklist_narrows_in_sql_not_in_the_page():
    """306 requests over seven days against a hundred-row window. A page that filtered
    what it had already fetched would report "12 refusals" when the window happened to
    hold 12 of 122 — the same lie as the count that used to read "100 requests". So
    the predicate has to sit above the LIMIT."""
    from src.agents.audit import RECENT_REQUESTS

    where = RECENT_REQUESTS.index("WHERE question.event_type")
    limit = RECENT_REQUESTS.index("LIMIT")
    for bound in ("outcome", "like", "user", "user_id", "user_like", "document", "document_like"):
        assert f"%({bound})s" in RECENT_REQUESTS[where:limit], f"{bound} is applied after the LIMIT"

    assert _ran(outcome="refused")["params"]["outcome"] == "refused"


def test_an_unknown_outcome_shows_everything_rather_than_nothing():
    """A typo in a filter must not silently hide rows from an audit surface. There is
    no safe way to guess what someone meant, and the honest fallback is everything."""
    assert _ran(outcome="refsued")["params"]["outcome"] == "all"
    assert _ran(outcome="")["params"]["outcome"] == "all"


def test_a_search_for_a_wildcard_is_a_search_for_that_wildcard():
    """The % and _ in a caller's text are theirs, not LIKE's. Unescaped, searching
    "50%" matches every row, which on this page reads as "your filter found
    everything" rather than "your filter did nothing"."""
    assert _ran(q="50%")["params"]["like"] == r"%50\%%"
    assert _ran(q="a_b")["params"]["like"] == r"%a\_b%"
    assert _ran(q="  spaced  ")["params"]["q"] == "spaced"
    assert _ran(q="")["params"]["q"] == ""


def test_a_user_is_found_by_id_or_by_part_of_a_name_or_email():
    """All digits is an id and nothing else: "2" must not also find everyone with a 2 in
    their address. Anything else is a fragment of a name or an email."""
    assert _ran(user="2")["params"]["user_id"] == 2
    assert _ran(user=" 17 ")["params"]["user_id"] == 17
    for text in ("marcus", "alex.tan@", "2fa"):
        assert _ran(user=text)["params"]["user_id"] is None
    assert _ran(user="marcus")["params"]["user_like"] == "%marcus%"
    assert _ran(user="")["params"]["user"] == ""


def test_a_document_filter_takes_its_wildcards_literally():
    assert _ran(document="confluence:SUPPORT/")["params"]["document_like"] == "%confluence:SUPPORT/%"
    assert _ran(document="50%")["params"]["document_like"] == r"%50\%%"
    assert _ran(document="a_b")["params"]["document_like"] == r"%a\_b%"
    assert _ran(document="  space  ")["params"]["document"] == "space"


def test_the_document_filter_matches_extracted_refs_never_payload_text():
    """An answer that merely MENTIONS a document has not touched it. The names come from
    four structured places, so free text in an answer can never make a request match."""
    from src.agents.audit import NAMED_DOCUMENTS

    for source in ("'citations'", "'conflicts'", "'items'", "'sources'"):
        assert f"payload->{source}" in NAMED_DOCUMENTS
    assert "payload::text" not in NAMED_DOCUMENTS and "->>'text'" not in NAMED_DOCUMENTS


def test_the_filters_reach_the_query_from_the_route():
    """GET /audit/recent?user=...&document=... — the officer's inquiry end to end, minus the
    database. `user` is a query parameter here even though the officer dependency is also
    called user on the Python side."""
    seen: dict = {}

    def connect():
        return _FakeConn(seen)

    from src.api.compliance import build_router
    from src.api.auth import build_dependencies
    from fastapi import FastAPI

    _, officer = build_dependencies(lambda uid: MARCUS)
    app = FastAPI()
    app.include_router(build_router(lambda uid: MARCUS, connect=connect, officer=officer))
    r = TestClient(app).get(
        "/audit/recent?days=30&user=jdoe&document=confluence:PAY/&outcome=refused",
        headers=as_user(2),
    )
    assert r.status_code == 200
    params = seen["params"]
    assert (params["user"], params["document"], params["outcome"]) == ("jdoe", "confluence:PAY/", "refused")
    assert params["document_like"] == "%confluence:PAY/%"
