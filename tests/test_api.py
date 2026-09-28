"""POST /query. The trust boundary: identity is resolved server-side, and the
response cannot carry the reason for a refusal.
"""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import get_settings
from src.graph.state import (
    GENERIC_REFUSAL,
    AuditEvent,
    Citation,
    PermConflict,
    UserContext,
    VerificationResult,
)
from tests.helpers import as_user


@pytest.fixture(autouse=True)
def pinned_settings(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MAX_HOPS", "3")
    monkeypatch.setenv("VERIFIER_CONFIDENCE_THRESHOLD", "0.6")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)


def _citation() -> Citation:
    return Citation(
        document_id=1,
        title="Customer Refund & Chargeback Policy",
        source_platform="confluence",
        source_ref="SUPPORT/refund-policy",
    )


def _nodes(**overrides):
    base = {
        "router": lambda s: {"route": "rag"},
        "retrieval": lambda s: {"hop_count": s.get("hop_count", 0) + 1},
        "sql_tool": lambda s: {"sql_result": None},
        "clarification": lambda s: {"clarification_question": "Which ticket?"},
        "synthesizer": lambda s: {"draft_answer": "45 days.", "citations": [_citation()]},
        "verifier": lambda s: {
            "verification": VerificationResult(grounded=True, unsupported=[], confidence=0.9)
        },
        "escalation": lambda s: {"escalated": True, "explanation": "matched a Compliance-only page"},
        "audit": lambda s: {
            "audit_events": [
                AuditEvent(event_type="final_answer", payload={}, occurred_at=datetime.now(UTC))
            ]
        },
    }
    base.update(overrides)
    return base


def _client(**overrides) -> TestClient:
    return TestClient(create_app(nodes=_nodes(**overrides), user_loader=lambda uid: ALEX))


def test_answers_a_permitted_question_with_citations():
    r = _client().post("/query", json={"query": "How long to contest a chargeback?"}, headers=as_user(1))
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "45 days."
    assert body["citations"][0]["source_ref"] == "SUPPORT/refund-policy"


def test_refusal_body_contains_no_reason():
    """Scenario 3 at the HTTP boundary — the wire format itself must not leak."""
    r = _client(
        retrieval=lambda s: {
            "hop_count": 1,
            "permission_conflicts": [
                PermConflict(
                    document_id=2,
                    source_platform="confluence",
                    source_ref="COMPLIANCE/aml-escalation",
                    sensitivity="restricted",
                    score_margin=0.2,
                )
            ],
        }
    ).post("/query", json={"query": "What triggers an AML escalation?"}, headers=as_user(1))

    assert r.status_code == 200
    body = r.json()
    assert body["text"] == GENERIC_REFUSAL
    assert body["citations"] == []
    # A tripwire on the shape, so a field added to AskerResponse has to be considered
    # here before it can ride out on a refusal. `trace` joined it when the UI grew a
    # pipeline view: empty on every refusal, because a trace reconstructs the cause
    # the wording withholds — a permission conflict stops at retrieval on hop 1, an
    # unsupported answer grinds through three.
    assert set(body) == {"text", "citations", "trace"}, f"unexpected keys: {set(body)}"
    assert body["trace"] == []

    serialised = r.text.lower()
    for leak in ("explanation", "aml", "compliance", "restricted", "conflict"):
        assert leak not in serialised


def test_caller_cannot_supply_its_own_role_or_clearance():
    """Accepting any of these from the body would walk straight past the §1 filter.

    `user_id` is on this list now too: it was a real field until the caller came from
    a signed token, and it was never checked against anything."""
    r = _client().post(
        "/query",
        json={"query": "anything", "user_id": 1, "role": "compliance", "clearance_level": 1},
        headers=as_user(1),
    )
    assert r.status_code == 422


def test_a_token_naming_nobody_is_rejected_without_saying_so():
    """401, not 404. A 404 answered the question "does user 999 exist?", which is a
    free account enumeration for anyone holding any valid signing key."""
    app = create_app(nodes=_nodes(), user_loader=lambda uid: None)
    r = TestClient(app).post("/query", json={"query": "anything"}, headers=as_user(999))
    assert r.status_code == 401
    assert "999" not in r.text and "user" not in r.json()["detail"].lower()


def test_empty_query_is_rejected():
    assert _client().post("/query", json={"query": ""}, headers=as_user(1)).status_code == 422


def test_identity_used_downstream_is_the_loaded_one_not_the_request():
    seen = {}

    def capture(state):
        seen["user"] = state["user"]
        return {"route": "rag"}

    client = TestClient(
        create_app(nodes=_nodes(router=capture), user_loader=lambda uid: ALEX)
    )
    client.post("/query", json={"query": "anything"}, headers=as_user(1))
    assert seen["user"].role == "support"
    assert seen["user"].acl_tags() == ["support", "support", "all-staff"]


# --- startup warm-up -----------------------------------------------------------

@pytest.mark.asyncio
async def test_warm_up_does_nothing_when_it_is_off():
    """conftest.py turns it off for the suite: every create_app would otherwise load
    a 7B model to answer nothing."""
    from src.api.main import _warm
    from src.config import Settings

    await _warm(Settings(_env_file=None, warm_on_startup=False))  # must not raise


@pytest.mark.asyncio
async def test_a_failed_warm_up_is_a_slow_first_answer_not_a_broken_start(monkeypatch):
    """The server is already serving when this runs. A model that will not load is
    a cold first question, never a failed startup."""
    from src.api import main
    from src.config import Settings

    def unreachable():
        raise ConnectionError("ollama is not running")

    monkeypatch.setattr(main, "run_in_threadpool", lambda fn: unreachable())
    await main._warm(Settings(_env_file=None, warm_on_startup=True))  # must not raise


def test_an_answer_carries_its_trace_and_a_refusal_does_not():
    """The pipeline the UI replays. On an answer it is what the graph did; on a
    refusal it is empty, because its shape names the cause the sentence withholds."""
    from src.graph.graph import build_response
    from src.graph.state import AuditEvent

    ran = [
        AuditEvent(event_type="node_transition",
                   payload={"node": "retrieval", "hop": 1, "ms": 48.3},
                   occurred_at=datetime.now(UTC)),
        AuditEvent(event_type="node_transition",
                   payload={"node": "verifier", "hop": 1, "ms": 2361.9},
                   occurred_at=datetime.now(UTC)),
    ]
    answered = build_response(
        {"escalated": False, "final_answer": "45 days.", "citations": [], "audit_events": ran}
    )
    assert [s.node for s in answered.trace] == ["retrieval", "verifier"]
    assert answered.trace[1].ms == 2361.9

    refused = build_response(
        {"escalated": True, "final_answer": "45 days.", "citations": [], "audit_events": ran}
    )
    assert refused.trace == []


def test_a_cache_hit_reports_the_cache_not_the_run_it_replays():
    """The cached response carries the trace of the request that FILLED the cache.
    Passing it through would show seconds of retrieval and synthesis that did not
    happen on this request — a plausible, wrong number next to a 0.1s clock."""
    from src.graph.state import AskerResponse, Stage

    stored = AskerResponse(
        text="45 days.",
        citations=[_citation()],
        trace=[Stage(node="synthesizer", hop=1, ms=2704.4)],
    )

    class AlwaysHits:
        def get(self, query, user):
            return stored

        def put(self, query, user, response):
            raise AssertionError("a hit must not re-cache")

    app = create_app(nodes=_nodes(), user_loader=lambda uid: ALEX, cache=AlwaysHits())
    body = TestClient(app).post("/query", json={"query": "anything"}, headers=as_user(1)).json()

    assert body["text"] == "45 days."
    assert [s["node"] for s in body["trace"]] == ["cache"]
    assert stored.trace[0].node == "synthesizer", "the cached entry was mutated"
