"""LangGraph wiring for the request-time flow (CLAUDE.md §2.2).

Nodes are injected rather than imported so the topology can be built and tested
before the retrieval and audit workstreams land their modules. Each node takes
`GraphState` and returns the state fragment it owns.
"""

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from src.config import get_settings
from src.graph.state import GENERIC_REFUSAL, AskerResponse, GraphState

Node = Callable[[GraphState], dict]


def answer_node(state: GraphState) -> dict:
    """Assemble what the asker receives.

    Re-forces the generic refusal when escalated. Escalation already writes it
    (§4), but this is the last node before the answer leaves the graph and §5 is
    the requirement the brief's negative case is built to break — so it is
    enforced here too rather than trusted upstream.
    """
    if state.get("escalated"):
        return {"final_answer": GENERIC_REFUSAL, "citations": []}

    # An ambiguous question ends the turn with the question itself. Checked after
    # the refusal, never before it: a clarification must not become a side channel
    # that distinguishes "nothing found" from "not permitted" (§5).
    if state.get("clarification_question") and not state.get("draft_answer"):
        return {"final_answer": state["clarification_question"], "citations": []}

    return {
        "final_answer": state.get("draft_answer"),
        "citations": state.get("citations") or [],
    }


def build_response(state: GraphState) -> AskerResponse:
    """State to the asker-facing payload (§5). Called at the API boundary."""
    if state.get("escalated"):
        return AskerResponse(text=GENERIC_REFUSAL)
    return AskerResponse(
        text=state.get("final_answer") or "",
        citations=state.get("citations") or [],
    )


def _from_router(state: GraphState) -> str:
    return state["route"]


def _from_retrieval(state: GraphState) -> str:
    # A restricted item outscored everything permitted — refuse before anything is
    # drafted, so no restricted content ever reaches the Synthesizer's prompt.
    if state.get("permission_conflicts"):
        return "escalation"
    # A retry that found exactly the evidence the last hop judged insufficient can
    # only reach the same verdict. Drafting and verifying it again cost two model
    # calls a hop and pushed these refusals to 10s, past the refusal deadline.
    if state.get("evidence_exhausted"):
        return "escalation"
    return "synthesizer"


def _noting_repeats(retrieval: Node) -> Node:
    """Wrap Retrieval to flag a hop that returned the same chunks as the one before."""

    def node(state: GraphState) -> dict:
        out = retrieval(state)
        before = {c.id for c in state.get("retrieved_chunks") or []}
        after = {c.id for c in out.get("retrieved_chunks") or []}
        out["evidence_exhausted"] = state.get("hop_count", 0) > 0 and after == before
        return out

    return node


def _from_verifier(state: GraphState) -> str:
    settings = get_settings()
    verification = state.get("verification")
    if verification is None:
        return "escalation"

    # Checked before `grounded`: §4 sends a low-confidence answer to Escalation
    # even when the Verifier believes it is supported.
    if verification.confidence < settings.verifier_confidence_threshold:
        return "escalation"
    if verification.grounded:
        return "answer"
    if state.get("hop_count", 0) < settings.retrieval_max_hops:
        return "retrieval"
    return "escalation"


def build_graph(
    *,
    router: Node,
    retrieval: Node,
    sql_tool: Node,
    clarification: Node,
    synthesizer: Node,
    verifier: Node,
    escalation: Node,
    audit: Node,
    answer: Node = answer_node,
):
    g = StateGraph(GraphState)

    g.add_node("router", router)
    g.add_node("retrieval", _noting_repeats(retrieval))
    g.add_node("sql_tool", sql_tool)
    g.add_node("clarification", clarification)
    g.add_node("synthesizer", synthesizer)
    g.add_node("verifier", verifier)
    g.add_node("escalation", escalation)
    g.add_node("answer", answer)
    g.add_node("audit", audit)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        _from_router,
        {
            "rag": "retrieval",
            "sql": "sql_tool",
            "clarify": "clarification",
            "escalate": "escalation",
        },
    )
    # Ends the turn rather than looping to the Router: POST /query is one request and
    # one response, so the asker answers by asking again with a fuller question. That
    # also makes §4's "one round only" structural — the Router runs once per request.
    g.add_edge("clarification", "answer")
    g.add_conditional_edges(
        "retrieval", _from_retrieval, {"escalation": "escalation", "synthesizer": "synthesizer"}
    )
    # §4: a SQL result reaches the Verifier "exactly like a retrieved chunk", so both
    # evidence paths are drafted by the same node before judgement.
    g.add_edge("sql_tool", "synthesizer")
    g.add_edge("synthesizer", "verifier")
    g.add_conditional_edges(
        "verifier",
        _from_verifier,
        {"answer": "answer", "retrieval": "retrieval", "escalation": "escalation"},
    )
    g.add_edge("escalation", "answer")

    # ponytail: audit as one terminal node, matching the §2.2 diagram. §4 says "one
    # row per node transition", which this cannot produce — that needs a wrapper
    # around every node. Left for the audit workstream to decide before it writes
    # verify_audit_chain() against a row count it expects.
    g.add_edge("answer", "audit")
    g.add_edge("audit", END)

    return g.compile()
