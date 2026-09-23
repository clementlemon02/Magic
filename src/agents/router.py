"""Router / Planner — CLAUDE.md §4.

In: `query`, `user`. Out: `route`.

Few-shot classification into rag | sql | clarify. Compound queries take their
dominant intent; the MVP does not split sub-queries.
"""

from typing import get_args

from src.agents.router_examples import EXAMPLES
from src.graph.state import GraphState, Route

VALID_ROUTES: tuple[str, ...] = get_args(Route)

# `escalate` is reachable only from Retrieval and the Verifier, never chosen here —
# the Router sees the question, not the evidence it takes to justify a refusal.
SELECTABLE_ROUTES = ("rag", "sql", "clarify")

PROMPT_TEMPLATE = """You route an employee's question to one handler at Aurelia Financial.

Answer with exactly one word, one of: rag, sql, clarify.

  rag      — the answer lives in written material: policies, procedures, incident
             write-ups, chat threads, standards.
  sql      — the answer is a number computed over the transactions table: counts,
             sums, averages, and anything scoped by date or department.
  clarify  — the question has no resolvable subject, and neither a document nor a
             table could be chosen without guessing.

A question that mixes intents takes its DOMINANT one. Do not split it.

Examples:
{examples}

Question: {query}
Answer with one word only."""


def _render_examples() -> str:
    return "\n".join(f"Question: {e['query']}\nAnswer: {e['route']}" for e in EXAMPLES)


def _parse(raw: str) -> str:
    """Pull a route out of the model's reply.

    Falls back to `rag` rather than raising: rag is the safe default because every
    chunk it can reach is still ACL-filtered in SQL (§1), so a misroute cannot leak.
    Defaulting to `clarify` would stall a valid question, and `escalate` would
    refuse one.
    """
    word = raw.strip().lower().strip(".,:;!?\"'`*")
    if word in SELECTABLE_ROUTES:
        return word
    # Models sometimes answer in a sentence; take the first route mentioned.
    for token in word.replace("\n", " ").split():
        cleaned = token.strip(".,:;!?\"'`*")
        if cleaned in SELECTABLE_ROUTES:
            return cleaned
    return "rag"


def classify(query: str, chat_model=None) -> str:
    """Classify a single query. `chat_model` is injectable so tests need no credentials."""
    if chat_model is None:
        from src.llm.factory import get_chat_model

        chat_model = get_chat_model()

    prompt = PROMPT_TEMPLATE.format(examples=_render_examples(), query=query)
    reply = chat_model.invoke(prompt)
    return _parse(getattr(reply, "content", reply))


def route_node(state: GraphState, chat_model=None) -> dict:
    """Graph node. Returns the state fragment the Router owns."""
    # §4's "one round only" needs no guard here: Clarification ends the turn, so the
    # Router runs at most once per request.
    return {"route": classify(state["query"], chat_model=chat_model)}
