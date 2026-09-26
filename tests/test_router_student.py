"""The distilled Router's gate: confident -> no LLM call; unsure -> the LLM Router decides."""

import json

import pytest

from src.agents import router
from src.config import get_settings
from src.graph.state import UserContext

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)


class FakeEmbeddings:
    """Maps a question to a fixed 2-d vector."""

    def __init__(self, vectors):
        self.vectors = vectors

    def embed_query(self, text):
        return self.vectors[text]


class CountingChat:
    def __init__(self, reply="rag"):
        self.reply, self.calls = reply, 0

    def invoke(self, prompt):
        self.calls += 1
        return type("Reply", (), {"content": self.reply})()


@pytest.fixture
def student(tmp_path, monkeypatch):
    # Axis 0 means rag, axis 1 means sql; clarify never wins. Weights large enough
    # that an on-axis vector is near-certain and a diagonal one is a coin toss.
    path = tmp_path / "router_student.json"
    path.write_text(json.dumps({
        "labels": ["rag", "sql", "clarify"],
        "embedding_model": "mxbai-embed-large",
        "weights": [[20, 0, 0], [0, 20, 0]],
        "bias": [0, 0, -50],
    }))
    monkeypatch.setattr(router, "STUDENT_PATH", path)
    monkeypatch.setenv("ROUTER_STUDENT_ENABLED", "true")
    monkeypatch.setenv("ROUTER_STUDENT_MIN_CONFIDENCE", "0.9")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    get_settings.cache_clear()
    router._load_student.cache_clear()
    yield
    get_settings.cache_clear()
    router._load_student.cache_clear()


def _state(q):
    return {"query": q, "user": ALEX}


def test_a_confident_student_decides_without_calling_the_llm(student):
    chat = CountingChat(reply="clarify")
    emb = FakeEmbeddings({"How many payments in July?": [0.0, 1.0]})
    out = router.route_node(_state("How many payments in July?"), chat_model=chat, embeddings=emb)
    assert out == {"route": "sql"}
    assert chat.calls == 0


def test_an_unsure_student_hands_the_decision_to_the_llm_router(student):
    chat = CountingChat(reply="rag")
    emb = FakeEmbeddings({"Summarise the outage and count failures": [0.7, 0.7]})
    out = router.route_node(
        _state("Summarise the outage and count failures"), chat_model=chat, embeddings=emb
    )
    assert out == {"route": "rag"}
    assert chat.calls == 1


def test_a_student_trained_on_another_embedding_model_is_not_used(student, monkeypatch):
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    get_settings.cache_clear()
    router._load_student.cache_clear()
    assert router.student_probabilities("q", FakeEmbeddings({"q": [1.0, 0.0]})) is None


def test_probabilities_sum_to_one(student):
    p = router.student_probabilities("q", FakeEmbeddings({"q": [0.6, 0.8]}))
    assert set(p) == {"rag", "sql", "clarify"}
    assert sum(p.values()) == pytest.approx(1.0)
