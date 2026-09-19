"""Verifier / Critic — CLAUDE.md §4.

In: `draft_answer`, `retrieved_chunks`. Out: `verification`.

Claim-level LLM-as-judge. Every failure path here fails CLOSED: an unparseable or
missing judgement becomes ungrounded at zero confidence, which routes to
Escalation and a generic refusal. An answer must be positively shown to be
grounded to reach the asker.
"""

import json
import re

from src.graph.state import Chunk, GraphState, VerificationResult

PROMPT_TEMPLATE = """You check whether an answer is supported by its evidence.

Split the answer into individual factual claims. A claim is supported only if the
evidence states it. Plausible, well-known, or likely-true claims are UNSUPPORTED
unless the evidence says so.

Reply with JSON only, no prose and no code fences:
{{"grounded": <true|false>, "unsupported": ["<claim>", ...], "confidence": <0.0-1.0>}}

"grounded" is true only when "unsupported" is empty.
"confidence" is how certain you are of your own judgement, not of the answer.

Evidence:
{evidence}

Question: {query}
Answer to check: {answer}

JSON:"""

# Fail-closed result for anything we could not read as a judgement.
_UNREADABLE = VerificationResult(
    grounded=False,
    unsupported=["verifier response could not be parsed"],
    confidence=0.0,
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _render_evidence(chunks: list[Chunk], sql_result) -> str:
    parts = [f"[{i}] {c.content}" for i, c in enumerate(chunks, start=1)]
    if sql_result is not None:
        parts.append(f"[query result] {sql_result}")
    return "\n\n".join(parts) if parts else "(none)"


def _parse(raw: str) -> VerificationResult:
    text = _FENCE.sub("", str(raw)).strip()
    # Models often wrap the object in a sentence; take the outermost braces.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return _UNREADABLE
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return _UNREADABLE

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)):
        return _UNREADABLE
    # Asked for 0.0-1.0, judges sometimes answer on a 0-100 scale. That reading is
    # unambiguous; anything outside both ranges is not, so it fails closed.
    if 1 < confidence <= 100:
        confidence = confidence / 100
    if not 0 <= confidence <= 1:
        return _UNREADABLE

    unsupported = data.get("unsupported") or []
    if not isinstance(unsupported, list):
        return _UNREADABLE
    unsupported = [str(u) for u in unsupported]

    # Trust the list over the flag: a judge that names unsupported claims and still
    # says grounded=true is reporting an answer we must not ship.
    grounded = bool(data.get("grounded")) and not unsupported

    return VerificationResult(
        grounded=grounded, unsupported=unsupported, confidence=float(confidence)
    )


def verify(
    query: str,
    draft_answer: str | None,
    chunks: list[Chunk],
    sql_result=None,
    chat_model=None,
) -> VerificationResult:
    if not draft_answer:
        # The Synthesizer found the permitted evidence insufficient. Nothing to judge;
        # let the hop loop retry or the hop cap escalate.
        return VerificationResult(
            grounded=False, unsupported=["no answer was drafted from permitted evidence"], confidence=1.0
        )

    if chat_model is None:
        from src.llm.factory import get_chat_model

        chat_model = get_chat_model()

    prompt = PROMPT_TEMPLATE.format(
        evidence=_render_evidence(chunks, sql_result), query=query, answer=draft_answer
    )
    reply = chat_model.invoke(prompt)
    return _parse(getattr(reply, "content", reply))


def verify_node(state: GraphState, chat_model=None) -> dict:
    return {
        "verification": verify(
            state["query"],
            state.get("draft_answer"),
            state.get("retrieved_chunks") or [],
            sql_result=state.get("sql_result"),
            chat_model=chat_model,
        )
    }
