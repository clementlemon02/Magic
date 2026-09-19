"""Synthesizer — drafts an answer from evidence the caller is allowed to see.

NOT in CLAUDE.md §4. Added because nothing else writes `draft_answer`: Retrieval
outputs chunks, the SQL Tool outputs rows, and the Verifier takes `draft_answer`
as an input. This node fills that gap, for both the rag and sql paths — §4 says a
SQL result reaches the Verifier "exactly like a retrieved chunk", so both kinds of
evidence are drafted the same way here.

Needs a §4 contract entry and a §8 owner before code freeze.
"""

from src.graph.state import Chunk, Citation, GraphState

PROMPT_TEMPLATE = """Answer the employee's question using ONLY the evidence below.

Rules:
- Use nothing outside the evidence. No background knowledge, no inference beyond it.
- If the evidence does not answer the question, say exactly: INSUFFICIENT
- Be direct. No preamble, no restating the question.

Evidence:
{evidence}

Question: {query}
Answer:"""

INSUFFICIENT = "INSUFFICIENT"


def _render_evidence(chunks: list[Chunk], sql_result) -> str:
    parts = [f"[{i}] {c.content}" for i, c in enumerate(chunks, start=1)]
    if sql_result is not None:
        parts.append(f"[query result] {sql_result}")
    return "\n\n".join(parts) if parts else "(none)"


def _citations(chunks: list[Chunk]) -> list[Citation]:
    """One citation per source document, in the order the chunks scored."""
    seen: set[int] = set()
    out: list[Citation] = []
    for c in chunks:
        if c.citation.document_id not in seen:
            seen.add(c.citation.document_id)
            out.append(c.citation)
    return out


def synthesize(query: str, chunks: list[Chunk], sql_result=None, chat_model=None) -> tuple[str | None, list[Citation]]:
    if chat_model is None:
        from src.llm.factory import get_chat_model

        chat_model = get_chat_model()

    if not chunks and sql_result is None:
        # Nothing permitted matched. Let the Verifier record it as ungrounded so the
        # hop loop and the escalation path stay the only ways this ends.
        return None, []

    prompt = PROMPT_TEMPLATE.format(evidence=_render_evidence(chunks, sql_result), query=query)
    reply = chat_model.invoke(prompt)
    text = str(getattr(reply, "content", reply)).strip()

    if text.upper().startswith(INSUFFICIENT):
        return None, []
    return text, _citations(chunks)


def synthesize_node(state: GraphState, chat_model=None) -> dict:
    draft, citations = synthesize(
        state["query"],
        state.get("retrieved_chunks") or [],
        sql_result=state.get("sql_result"),
        chat_model=chat_model,
    )
    return {"draft_answer": draft, "citations": citations}
