"""Failure behaviour. A judged demo that crashes is unrecoverable, and a broken
dependency must never be reported in the same words as a withheld answer.
"""

import psycopg
import pytest
from fastapi.testclient import TestClient

from src.api.main import check_llm, create_app, describe_failure
from src.config import get_settings
from src.graph.state import GENERIC_REFUSAL, UserContext, VerificationResult

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)


def _nodes(**overrides):
    base = {
        "router": lambda s: {"route": "rag"},
        "retrieval": lambda s: {"hop_count": 1},
        "sql_tool": lambda s: {"sql_result": None},
        "clarification": lambda s: {"clarification_question": "Which one?"},
        "synthesizer": lambda s: {"draft_answer": "45 days.", "citations": []},
        "verifier": lambda s: {
            "verification": VerificationResult(grounded=True, unsupported=[], confidence=0.9)
        },
        "escalation": lambda s: {"escalated": True},
        "audit": lambda s: {"audit_events": []},
    }
    base.update(overrides)
    return base


def _client(**overrides) -> TestClient:
    app = create_app(nodes=_nodes(**overrides), user_loader=lambda uid: ALEX)
    # The app's own handler must produce the response, not the test client.
    return TestClient(app, raise_server_exceptions=False)


def test_database_fault_is_503_not_a_crash():
    def dies(state):
        raise psycopg.OperationalError("connection to server was lost")

    r = _client(retrieval=dies).post("/query", json={"query": "anything", "user_id": 1})
    assert r.status_code == 503
    assert r.json()["detail"] == "The knowledge store is unavailable."


def test_model_timeout_is_503_not_a_crash():
    def dies(state):
        raise TimeoutError("read timed out")

    r = _client(synthesizer=dies).post("/query", json={"query": "anything", "user_id": 1})
    assert r.status_code == 503
    assert "language model" in r.json()["detail"]


def test_unimplemented_node_is_501():
    """The retrieval, escalation and audit stubs, before those workstreams land."""

    def pending(state):
        raise NotImplementedError("retrieval node is not implemented yet (Chris)")

    r = _client(retrieval=pending).post("/query", json={"query": "anything", "user_id": 1})
    assert r.status_code == 501


def test_a_broken_dependency_never_looks_like_a_refusal():
    """The load-bearing one.

    If an outage produced the generic refusal, that sentence would mean three things —
    withheld, nothing found, and broken — and a judge would read a crash as a security
    feature. Worse, it weakens §5: the refusal stops being uninformative.
    """
    for boom in (psycopg.OperationalError("down"), TimeoutError("slow"), RuntimeError("?")):

        def dies(state, exc=boom):
            raise exc

        r = _client(retrieval=dies).post("/query", json={"query": "anything", "user_id": 1})
        assert r.status_code >= 500
        assert GENERIC_REFUSAL not in r.text
        assert "permitted" not in r.text.lower()


def test_a_fault_resolving_the_caller_is_handled_too():
    """Raised in a dependency, before the endpoint body runs."""

    def dies(user_id):
        raise psycopg.OperationalError("connection refused")

    app = create_app(nodes=_nodes(), user_loader=dies)
    r = TestClient(app, raise_server_exceptions=False).post(
        "/query", json={"query": "anything", "user_id": 1}
    )
    assert r.status_code == 503


def test_unknown_user_is_still_a_404_not_a_503():
    """HTTPException must survive the catch-all handler."""
    app = create_app(nodes=_nodes(), user_loader=lambda uid: None)
    r = TestClient(app, raise_server_exceptions=False).post(
        "/query", json={"query": "anything", "user_id": 999}
    )
    assert r.status_code == 404


@pytest.mark.parametrize(
    "exc,status",
    [
        (NotImplementedError(), 501),
        (psycopg.OperationalError(), 503),
        (TimeoutError(), 503),
        (ConnectionError(), 503),
        (ValueError(), 503),
    ],
)
def test_failure_classification(exc, status):
    code, message = describe_failure(exc)
    assert code == status
    assert message and GENERIC_REFUSAL not in message


def test_health_reports_each_dependency():
    r = _client().get("/health")
    assert r.status_code in (200, 503)
    assert set(r.json()["checks"]) == {"database", "llm"}


def test_health_is_degraded_when_the_llm_is_unreachable(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:1")  # nothing listens here
    get_settings.cache_clear()
    try:
        assert check_llm() == "unavailable"
    finally:
        get_settings.cache_clear()


def test_refusals_are_never_cached():
    """A refusal owes an escalations row (§4); serving one from memory would skip
    the graph that writes it. Only answers are cached.
    """
    from src.cache import PermissionAwareCache
    from src.graph.state import PermConflict

    class FakeEmb:
        def embed_query(self, text):
            return [1.0, 0.0]

    cache = PermissionAwareCache(embeddings=FakeEmb(), execute=lambda sql, p: [])
    conflict = PermConflict(
        document_id=7, source_platform="confluence", source_ref="COMPLIANCE/aml",
        sensitivity="restricted", score_margin=0.3,
    )
    app = create_app(
        nodes=_nodes(retrieval=lambda s: {"hop_count": 1, "permission_conflicts": [conflict]}),
        user_loader=lambda uid: ALEX,
        cache=cache,
    )
    client = TestClient(app, raise_server_exceptions=False)

    first = client.post("/query", json={"query": "what triggers aml review", "user_id": 1})
    assert first.json()["text"] == GENERIC_REFUSAL
    assert cache._entries == [], "a refusal was cached"

    # The second request must go through the graph again, not the cache.
    second = client.post("/query", json={"query": "what triggers aml review", "user_id": 1})
    assert second.json()["text"] == GENERIC_REFUSAL
    assert cache.hits == 0


def test_answers_are_cached_and_served():
    from src.cache import PermissionAwareCache

    class FakeEmb:
        def embed_query(self, text):
            return [1.0, 0.0]

    cache = PermissionAwareCache(embeddings=FakeEmb(), execute=lambda sql, p: [])
    client = TestClient(
        create_app(nodes=_nodes(), user_loader=lambda uid: ALEX, cache=cache),
        raise_server_exceptions=False,
    )
    client.post("/query", json={"query": "how long to review a refund", "user_id": 1})
    client.post("/query", json={"query": "how long to review a refund", "user_id": 1})
    assert cache.hits == 1
