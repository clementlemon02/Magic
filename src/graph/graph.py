"""LangGraph wiring for the request-time flow (CLAUDE.md §2.2).

Nodes are injected rather than imported so the topology can be built and tested
before the retrieval and audit workstreams land their modules. Each node takes
`GraphState` and returns the state fragment it owns.
"""

import time
import time
from collections.abc import Callable
from datetime import UTC, datetime

from langgraph.graph import END, START, StateGraph

from src.config import get_settings
from src.graph.state import (
    DECLINE_REPLY,
    GENERIC_REFUSAL,
    AskerResponse,
    AuditEvent,
    GraphState,
    Stage,
)

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

    # Not a question about company knowledge at all. Also after the refusal branch:
    # the Router picks this from the query text before any retrieval, so it can never
    # be reached by a request that touched the permission filter — but the ordering
    # is what makes that true rather than merely likely.
    if state.get("route") == "decline":
        return {"final_answer": DECLINE_REPLY, "citations": []}

    return {
        "final_answer": state.get("draft_answer"),
        "citations": state.get("citations") or [],
    }


def build_response(state: GraphState) -> AskerResponse:
    """State to the asker-facing payload (§5). Called at the API boundary.

    The refusal branch constructs GENERIC_REFUSAL and nothing else. That is the whole
    contract: anything added to AskerResponse can only ever reach an answer.
    """
    if state.get("escalated"):
        return AskerResponse(text=GENERIC_REFUSAL)
    return AskerResponse(
        text=state.get("final_answer") or "",
        citations=state.get("citations") or [],
        trace=[
            Stage(node=e.payload["node"], hop=e.payload["hop"], ms=e.payload["ms"],
                  detail=e.payload.get("detail"))
            for e in (state.get("audit_events") or [])
            if e.event_type == "node_transition"
        ],
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
    if state.get("hop_count", 0) < settings.retrieval_max_hops and _budget_left(state):
        return "retrieval"
    return "escalation"


def _budget_left(state: GraphState) -> bool:
    """Whether another hop may START. Bounds how long a refusal can take by construction.

    Refusals are padded to REFUSAL_DEADLINE_SECONDS, but a refusal that runs past the
    deadline can't be padded back, and its lateness says it was not a permission
    conflict. With no new hop begun after the budget, a refusal takes at most the
    budget plus one hop — so the deadline is a guarantee, not a percentile.
    """
    started = state.get("started_at")
    if started is None:
        return True
    return time.monotonic() - started < get_settings().retrieval_hop_budget_seconds


def _detail(name: str, out: dict) -> str | None:
    """A few words about what a node just did, for the pipeline view.

    Read from the node's own output, which is the only place it exists. PERMITTED
    evidence only: "8 passages" is what this caller got, and how many were filtered
    out is exactly the number §5 forbids — it would say restricted material exists
    without naming it. A refusal carries no trace at all, so nothing here can ever
    describe a withheld request (see `build_response`).
    """
    if name == "router":
        return out.get("route")
    if name == "retrieval":
        chunks = out.get("retrieved_chunks")
        return f"{len(chunks)} passage{'' if len(chunks) == 1 else 's'}" if chunks else None
    if name == "synthesizer":
        cited = out.get("citations")
        return f"{len(cited)} cited" if cited else None
    if name == "sql_tool":
        result = out.get("sql_result")
        return (result or {}).get("template") if isinstance(result, dict) else None
    if name == "verifier":
        v = out.get("verification")
        return f"{'grounded' if v.grounded else 'ungrounded'} {v.confidence:.2f}" if v else None
    return None


def _recorded(name: str, node: Node) -> Node:
    """Wrap a node so the graph writes one row per transition (§4).

    The trail used to be derived from final state by the terminal audit node, so a
    request that hopped three times recorded its last hop and a count. These rows are
    written as each node returns: every hop is on the record with its own timestamp
    and duration.

    A node's own state writes win — retrieval and escalation already append their own
    events — so this appends to whichever list came back rather than the one it was
    handed. `audit` is not wrapped: it writes the trail, so a row about it could never
    be in the trail it wrote.
    """

    def wrapped(state: GraphState) -> dict:
        started = time.monotonic()
        out = node(state)
        events = list(out.get("audit_events") or state.get("audit_events") or [])
        events.append(
            AuditEvent(
                event_type="node_transition",
                payload={
                    "node": name,
                    "hop": out.get("hop_count", state.get("hop_count", 0)),
                    "ms": round((time.monotonic() - started) * 1000, 1),
                    "detail": _detail(name, out),
                },
                occurred_at=datetime.now(UTC),
            )
        )
        return {**out, "audit_events": events}

    return wrapped


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

    for name, node in (
        ("router", router),
        # _noting_repeats stays inside: it compares this hop's chunks with the last
        # and belongs to retrieval, not to recording.
        ("retrieval", _noting_repeats(retrieval)),
        ("sql_tool", sql_tool),
        ("clarification", clarification),
        ("synthesizer", synthesizer),
        ("verifier", verifier),
        ("escalation", escalation),
        ("answer", answer),
    ):
        g.add_node(name, _recorded(name, node))
    # Unwrapped: it writes the trail, so it cannot appear in it.
    g.add_node("audit", audit)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        _from_router,
        {
            "rag": "retrieval",
            "sql": "sql_tool",
            "clarify": "clarification",
            # No node of its own: the reply is a constant, so `answer` writes it.
            "decline": "answer",
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
