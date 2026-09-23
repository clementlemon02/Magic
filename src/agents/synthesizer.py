"""Synthesizer — drafts an answer from evidence the caller is allowed to see.

NOT in CLAUDE.md §4. Added because nothing else writes `draft_answer`: Retrieval
outputs chunks, the SQL Tool outputs rows, and the Verifier takes `draft_answer`
as an input. This node fills that gap, for both the rag and sql paths — §4 says a
SQL result reaches the Verifier "exactly like a retrieved chunk", so both kinds of
evidence are drafted the same way here.

Needs a §4 contract entry and a §8 owner before code freeze.
"""

import re

from src.graph.state import Chunk, Citation, GraphState

PROMPT_TEMPLATE = """Answer the employee's question using ONLY the evidence below.

Rules:
- Use nothing outside the evidence. No background knowledge, no inference beyond it.
- If the evidence does not answer the question, say exactly: INSUFFICIENT
- Be direct. No preamble, no restating the question.

Reply in exactly this form, with the answer first and the sources on their own last
line naming only the evidence numbers your answer actually used:

<your answer>
SOURCES: 1, 3

Evidence:
{evidence}

Question: {query}
Answer:"""

INSUFFICIENT = "INSUFFICIENT"

# The "SOURCES: 1, 3" marker. Stripped from the answer before it reaches the asker —
# it is bookkeeping, not prose. Deliberately unanchored: qwen2.5 puts it on its own
# line most of the time but will also tack it onto the end of the last sentence, and
# a line-anchored pattern leaks the marker into the answer when it does. Only digits
# and separators are consumed, so prose after it survives.
_SOURCES_LINE = re.compile(r"SOURCES\s*:\s*([0-9,\s]*)", re.IGNORECASE)


def _render_evidence(chunks: list[Chunk], sql_result) -> str:
    parts = [f"[{i}] {c.content}" for i, c in enumerate(chunks, start=1)]
    if sql_result is not None:
        parts.append(f"[query result] {sql_result}")
    return "\n\n".join(parts) if parts else "(none)"


def _split_sources(text: str) -> tuple[str, list[int] | None]:
    """Separate the answer from its trailing SOURCES line.

    Returns the cleaned answer and the 1-based evidence numbers it claimed, or None
    when the model did not emit a usable line.
    """
    match = _SOURCES_LINE.search(text)
    if match is None:
        return text.strip(), None

    answer = (text[: match.start()] + text[match.end() :]).strip()
    numbers = [int(n) for n in re.findall(r"\d+", match.group(1))]
    return answer, numbers or None


def _looks_like_no_answer(answer: str) -> bool:
    """Empty or INSUFFICIENT — either way, nothing was drafted.

    An empty draft is a real failure mode, not a hypothetical: asked a question that
    only one chunk answered, qwen2.5 replied with the SOURCES line alone and no prose.
    Returning "" would hand the asker a blank response; returning None routes it back
    through the hop loop and Escalation like any other ungrounded attempt.
    """
    return not answer or answer.upper().startswith(INSUFFICIENT)


def _citations(chunks: list[Chunk], used: list[int] | None = None) -> list[Citation]:
    """One citation per source document the answer actually drew on.

    `used` holds 1-based indices into `chunks`, as rendered in the prompt. Out-of-range
    numbers are dropped. When the model names nothing usable, every retrieved chunk is
    cited: each passed RETRIEVAL_MIN_SCORE, and for Scenario 1 an extra citation costs
    less than a missing one.
    """
    if used:
        selected = [chunks[i - 1] for i in used if 1 <= i <= len(chunks)]
        chunks = selected or chunks

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

    answer, used = _split_sources(text)
    if _looks_like_no_answer(answer):
        return None, []
    return answer, _citations(chunks, used)


def synthesize_node(state: GraphState, chat_model=None) -> dict:
    draft, citations = synthesize(
        state["query"],
        state.get("retrieved_chunks") or [],
        sql_result=state.get("sql_result"),
        chat_model=chat_model,
    )
    return {"draft_answer": draft, "citations": citations}
