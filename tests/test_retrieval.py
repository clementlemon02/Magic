"""Safety tests for the retrieval node's two-search contract."""

import pytest

from src.agents.retrieval import (
    FILTERED_SEARCH,
    RESTRICTED_CONFLICT_SEARCH,
    retrieve,
    retrieval_node,
)
from src.config import get_settings
from src.graph.state import UserContext, VerificationResult


class FakeEmbeddings:
    def __init__(self):
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2]


def _user() -> UserContext:
    return UserContext(id=9, role="support", dept="support", clearance_level=0)


def _permitted_row(score: float = 0.88) -> dict:
    return {
        "id": 10,
        "document_id": 1,
        "content": "Refund requests are reviewed within five business days.",
        "acl_tags": ["support"],
        "title": "Refund Policy",
        "source_platform": "confluence",
        "source_ref": "SUPPORT/refund-policy",
        "score": score,
    }


def _restricted_row(score: float = 0.96) -> dict:
    return {
        "document_id": 2,
        "source_platform": "confluence",
        "source_ref": "COMPLIANCE/aml-escalation",
        "sensitivity": "restricted",
        "score": score,
    }


def test_filtered_search_requires_live_permission_and_acl_metadata():
    assert "JOIN permissions AS permission" in FILTERED_SEARCH
    assert "permission.revoked_at IS NULL" in FILTERED_SEARCH
    assert "chunk.acl_tags && %(acl_tags)s" in FILTERED_SEARCH


def test_conflict_search_never_selects_chunk_content():
    assert "chunk.content" not in RESTRICTED_CONFLICT_SEARCH
    assert "NOT EXISTS" in RESTRICTED_CONFLICT_SEARCH


def test_retrieve_returns_only_permitted_content():
    calls = []

    def execute(sql, params):
        calls.append((sql, params))
        return [_permitted_row()] if sql == FILTERED_SEARCH else []

    chunks, conflicts = retrieve(
        "What is the refund policy?",
        _user(),
        execute,
        FakeEmbeddings(),
        top_k=6,
        min_score=0.72,
        conflict_score_margin=0.05,
    )

    assert chunks[0].content.startswith("Refund requests")
    assert chunks[0].citation.source_ref == "SUPPORT/refund-policy"
    assert conflicts == []
    assert len(calls) == 2
    assert calls[0][1]["acl_tags"] == ["support", "support", "user:9"]


def test_higher_ranked_restricted_match_creates_metadata_only_conflict():
    def execute(sql, params):
        return [_permitted_row(0.88)] if sql == FILTERED_SEARCH else [_restricted_row(0.96)]

    chunks, conflicts = retrieve(
        "What is the AML escalation procedure?",
        _user(),
        execute,
        FakeEmbeddings(),
        top_k=6,
        min_score=0.72,
        conflict_score_margin=0.05,
    )

    assert len(chunks) == 1
    assert conflicts[0].source_ref == "COMPLIANCE/aml-escalation"
    assert "content" not in conflicts[0].model_fields
    assert conflicts[0].score_margin == pytest.approx(0.08)


def test_restricted_result_within_margin_is_not_a_conflict():
    def execute(sql, params):
        return [_permitted_row(0.88)] if sql == FILTERED_SEARCH else [_restricted_row(0.93)]

    _, conflicts = retrieve(
        "Refund policy",
        _user(),
        execute,
        FakeEmbeddings(),
        top_k=6,
        min_score=0.72,
        conflict_score_margin=0.05,
    )

    assert conflicts == []


def test_restricted_match_without_permitted_evidence_is_a_conflict():
    def execute(sql, params):
        return [] if sql == FILTERED_SEARCH else [_restricted_row()]

    _, conflicts = retrieve(
        "AML escalation",
        _user(),
        execute,
        FakeEmbeddings(),
        top_k=6,
        min_score=0.72,
        conflict_score_margin=0.05,
    )

    assert len(conflicts) == 1


@pytest.fixture(autouse=True)
def retrieval_settings(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "6")
    monkeypatch.setenv("RETRIEVAL_MIN_SCORE", "0.72")
    monkeypatch.setenv("PERMISSION_CONFLICT_SCORE_MARGIN", "0.05")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_retrieval_node_reformulates_on_verifier_feedback_and_increments_hop():
    embeddings = FakeEmbeddings()

    def execute(sql, params):
        return [_permitted_row()] if sql == FILTERED_SEARCH else []

    result = retrieval_node(
        {
            "query": "What is the refund window?",
            "user": _user(),
            "hop_count": 1,
            "verification": VerificationResult(
                grounded=False, unsupported=["time limit"], confidence=0.8
            ),
        },
        embeddings=embeddings,
        execute=execute,
    )

    assert result["hop_count"] == 2
    assert result["permission_conflicts"] == []
    assert "Find evidence for: time limit" in embeddings.queries[0]
