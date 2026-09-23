"""Clarification — CLAUDE.md §4.

In: `query`. Out: `clarification_question`.

One round only. The Router enforces the cap (a second `clarify` becomes `rag`);
this module just writes the question.
"""

from src.graph.state import GraphState

PROMPT_TEMPLATE = """An employee asked a question that cannot be answered as written,
because it does not say what it is about.

Ask ONE short question that would let you resolve it. Ask for the missing subject —
which document, which ticket, which period, which system. Do not offer options you
are guessing at, and do not apologise.

Question: {query}
Your one clarifying question:"""

FALLBACK_QUESTION = "Which system, document, or time period should I look at?"


def clarify(query: str, chat_model=None) -> str:
    if chat_model is None:
        from src.llm.factory import get_chat_model

        chat_model = get_chat_model()

    reply = chat_model.invoke(PROMPT_TEMPLATE.format(query=query))
    question = str(getattr(reply, "content", reply)).strip()
    # An empty or rambling reply would strand the user mid-conversation; one round
    # is all we get (§4), so spend it on something answerable.
    if not question or len(question) > 300:
        return FALLBACK_QUESTION
    return question


def clarify_node(state: GraphState, chat_model=None) -> dict:
    return {"clarification_question": clarify(state["query"], chat_model=chat_model)}
