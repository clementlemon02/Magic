"""Scenarios 1 and 3 end to end, over HTTP, through the real agent code.

Router, Synthesizer, Verifier and the graph are the production implementations;
only the chat model and the two workstreams that have not merged yet (retrieval,
escalation, audit) are stood in for.
"""

from datetime import UTC, datetime
from functools import partial

import pytest
from fastapi.testclient import TestClient

from src.agents.clarification import clarify_node
from src.agents.router import route_node
from src.agents.sql_tool import sql_tool_node
from src.agents.synthesizer import synthesize_node
from src.agents.verifier import verify_node
from src.api.main import create_app
from src.config import get_settings
from src.graph.state import (
    GENERIC_REFUSAL,
    AuditEvent,
    Chunk,
    Citation,
    PermConflict,
    UserContext,
)
from src.llm.fake import FakeChatModel

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)


@pytest.fixture(autouse=True)
def pinned_settings(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MAX_HOPS", "3")
    monkeypatch.setenv("VERIFIER_CONFIDENCE_THRESHOLD", "0.6")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _chunk(doc_id, platform, ref, title, content) -> Chunk:
    return Chunk(
        id=doc_id * 10,
        document_id=doc_id,
        content=content,
        acl_tags=["support"],
        score=0.9,
        citation=Citation(
            document_id=doc_id, title=title, source_platform=platform, source_ref=ref
        ),
    )


def _real_nodes(fake_chat, *, retrieval, escalation, audit) -> dict:
    """Production agents, bound to a controlled chat model."""
    return {
        "router": partial(route_node, chat_model=fake_chat),
        "clarification": partial(clarify_node, chat_model=fake_chat),
        "synthesizer": partial(synthesize_node, chat_model=fake_chat),
        "verifier": partial(verify_node, chat_model=fake_chat),
        "sql_tool": partial(sql_tool_node, chat_model=fake_chat),
        "retrieval": retrieval,
        "escalation": escalation,
        "audit": audit,
    }


def test_scenario_1_answer_spans_two_platforms_and_cites_both():
    """Confluence policy page + Slack thread, each citation resolving to its own platform."""
    chat = FakeChatModel(
        route="rag",
        answer="Customers have 45 days to contest a chargeback, and on-call follows the PII process agreed in #support-eng.",
        grounded=True,
        confidence=0.92,
    )

    def retrieval(state):
        return {
            "hop_count": state.get("hop_count", 0) + 1,
            "retrieved_chunks": [
                _chunk(
                    1,
                    "confluence",
                    "SUPPORT/refund-policy",
                    "Customer Refund & Chargeback Policy",
                    "Chargebacks must be contested within 45 days.",
                ),
                _chunk(
                    2,
                    "slack",
                    "#support-eng/p1726",
                    "#support-eng thread",
                    "On-call PII access goes through the standing approval process.",
                ),
            ],
        }

    nodes = _real_nodes(
        chat,
        retrieval=retrieval,
        escalation=lambda s: {"escalated": True},
        audit=lambda s: {"audit_events": []},
    )
    client = TestClient(create_app(nodes=nodes, user_loader=lambda uid: ALEX))
    r = client.post("/query", json={"query": "How long to contest a chargeback?", "user_id": 1})

    assert r.status_code == 200
    body = r.json()
    assert "45 days" in body["text"]

    platforms = {c["source_platform"] for c in body["citations"]}
    assert platforms == {"confluence", "slack"}
    refs = {c["source_ref"] for c in body["citations"]}
    assert refs == {"SUPPORT/refund-policy", "#support-eng/p1726"}


def test_scenario_3_restricted_match_refuses_without_revealing_anything():
    """The reason reaches the audit log. It never reaches the asker."""
    chat = FakeChatModel(route="rag")
    audited = {}

    def retrieval(state):
        # The restricted page outscored everything Alex may see. Identity only.
        return {
            "hop_count": state.get("hop_count", 0) + 1,
            "retrieved_chunks": [],
            "permission_conflicts": [
                PermConflict(
                    document_id=7,
                    source_platform="confluence",
                    source_ref="COMPLIANCE/aml-escalation",
                    sensitivity="restricted",
                    score_margin=0.31,
                )
            ],
        }

    def escalation(state):
        conflict = state["permission_conflicts"][0]
        return {
            "escalated": True,
            "explanation": (
                f"Top match {conflict.source_ref} is {conflict.sensitivity}; "
                f"caller clearance {state['user'].clearance_level} is insufficient."
            ),
        }

    def audit(state):
        audited["explanation"] = state.get("explanation")
        audited["conflicts"] = state.get("permission_conflicts")
        return {
            "audit_events": [
                AuditEvent(
                    event_type="permission_conflict", payload={}, occurred_at=datetime.now(UTC)
                )
            ]
        }

    nodes = _real_nodes(chat, retrieval=retrieval, escalation=escalation, audit=audit)
    client = TestClient(create_app(nodes=nodes, user_loader=lambda uid: ALEX))
    r = client.post(
        "/query", json={"query": "What triggers an AML escalation review?", "user_id": 1}
    )

    assert r.status_code == 200
    assert r.json()["text"] == GENERIC_REFUSAL
    assert r.json()["citations"] == []

    # Nothing about the restricted item survives into the response, including the
    # words from the question itself being echoed back as confirmation.
    for leak in ("aml", "compliance", "restricted", "clearance", "escalat", "insufficient"):
        assert leak not in r.text.lower(), f"{leak!r} leaked into the response"

    # The specific reason exists — just not out there.
    assert "COMPLIANCE/aml-escalation" in audited["explanation"]
    assert audited["conflicts"][0].sensitivity == "restricted"


def test_scenario_3_never_puts_restricted_content_in_a_prompt():
    """The conflict check runs before synthesis, so no LLM ever sees the restricted item."""
    chat = FakeChatModel(route="rag")

    def retrieval(state):
        return {
            "hop_count": 1,
            "permission_conflicts": [
                PermConflict(
                    document_id=7,
                    source_platform="confluence",
                    source_ref="COMPLIANCE/aml-escalation",
                    sensitivity="restricted",
                    score_margin=0.31,
                )
            ],
        }

    nodes = _real_nodes(
        chat,
        retrieval=retrieval,
        escalation=lambda s: {"escalated": True, "explanation": "restricted"},
        audit=lambda s: {"audit_events": []},
    )
    TestClient(create_app(nodes=nodes, user_loader=lambda uid: ALEX)).post(
        "/query", json={"query": "What triggers an AML escalation review?", "user_id": 1}
    )

    # Only the Router was ever prompted; the Synthesizer and Verifier never ran.
    assert len(chat.prompts) == 1
    assert "Answer with exactly one word" in chat.prompts[0]
    for prompt in chat.prompts:
        assert "aml-escalation" not in prompt.lower()


def test_a_restricted_refusal_is_indistinguishable_from_having_no_answer():
    """The oracle property, and the reason the refusal is deliberately uninformative.

    A caller must not be able to tell "there is an answer you may not see" apart from
    "there is no answer". If the two responses differ by so much as a word, asking
    becomes a way to enumerate what restricted material exists — which is how the
    prompt-based baseline in evals/leak_probe.py discloses the corpus on 6 probes out
    of 6 while never quoting a restricted document.
    """
    chat = FakeChatModel(route="rag", grounded=False, unsupported=["nothing found"], confidence=0.9)

    def restricted_retrieval(state):
        return {
            "hop_count": state.get("hop_count", 0) + 1,
            "permission_conflicts": [
                PermConflict(
                    document_id=7,
                    source_platform="confluence",
                    source_ref="COMPLIANCE/aml-escalation",
                    sensitivity="restricted",
                    score_margin=0.31,
                )
            ],
        }

    def empty_retrieval(state):
        return {"hop_count": state.get("hop_count", 0) + 1, "retrieved_chunks": []}

    def client(retrieval):
        nodes = _real_nodes(
            chat,
            retrieval=retrieval,
            escalation=lambda s: {"escalated": True, "explanation": "internal reason"},
            audit=lambda s: {"audit_events": []},
        )
        return TestClient(create_app(nodes=nodes, user_loader=lambda uid: ALEX))

    restricted = client(restricted_retrieval).post(
        "/query", json={"query": "What triggers an AML escalation review?", "user_id": 1}
    )
    unanswerable = client(empty_retrieval).post(
        "/query", json={"query": "What is our parental leave policy?", "user_id": 1}
    )

    assert restricted.status_code == unanswerable.status_code == 200
    assert restricted.json() == unanswerable.json(), (
        "restricted and unanswerable responses differ — the response is an oracle"
    )
    assert restricted.json()["text"] == GENERIC_REFUSAL
