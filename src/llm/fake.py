"""Offline chat model, so the graph runs end to end without Hunyuan credentials.

Every agent's prompt carries a distinctive line, so this picks a canned reply by
matching on it. Replies are shaped like the real thing — a bare route word, a
verifier JSON object, a SQL plan — so the parsing and routing code under test is
the same code that runs in production.

This is a test and local-development backend. It is never selected by default;
`LLM_BACKEND=fake` has to be set on purpose, and doing so warns.
"""

import json
import warnings

# ponytail: dispatch by matching a line of each PROMPT_TEMPLATE. Cheap, and it
# breaks loudly (UnknownPrompt) rather than silently if a template is reworded.
# If prompts start sharing phrasing, give each template an explicit marker line.
_MARKERS = {
    "Answer with exactly one word": "router",
    "Ask ONE short question": "clarification",
    "Answer the employee's question using ONLY the evidence": "synthesizer",
    "You check whether an answer is supported": "verifier",
    "Pick the query that answers the question": "sql_tool",
}


class UnknownPrompt(RuntimeError):
    """A prompt this fake does not recognise — usually a reworded template."""


class _Reply:
    def __init__(self, content: str):
        self.content = content


class FakeChatModel:
    """Deterministic stand-in. Configure per-agent replies; defaults are sensible."""

    def __init__(
        self,
        *,
        route: str = "rag",
        answer: str = "The chargeback contest window is 45 days.",
        grounded: bool = True,
        unsupported: list[str] | None = None,
        confidence: float = 0.9,
        clarification: str = "Which ticket or document do you mean?",
        sql_plan: dict | None = None,
    ):
        self.route = route
        self.answer = answer
        self.grounded = grounded
        self.unsupported = unsupported or []
        self.confidence = confidence
        self.clarification = clarification
        self.sql_plan = sql_plan
        self.prompts: list[str] = []

    def _agent_for(self, prompt: str) -> str:
        for marker, agent in _MARKERS.items():
            if marker in prompt:
                return agent
        raise UnknownPrompt(f"no marker matched; prompt begins: {prompt[:80]!r}")

    def invoke(self, prompt: str) -> _Reply:
        self.prompts.append(prompt)
        agent = self._agent_for(prompt)

        if agent == "router":
            return _Reply(self.route)
        if agent == "clarification":
            return _Reply(self.clarification)
        if agent == "synthesizer":
            return _Reply(self.answer)
        if agent == "verifier":
            return _Reply(
                json.dumps(
                    {
                        "grounded": self.grounded and not self.unsupported,
                        "unsupported": self.unsupported,
                        "confidence": self.confidence,
                    }
                )
            )
        return _Reply(json.dumps(self.sql_plan) if self.sql_plan else '{"template": null}')


def warn_fake_backend() -> None:
    warnings.warn(
        "LLM_BACKEND=fake — answers are canned. Never record a demo on this backend.",
        RuntimeWarning,
        stacklevel=2,
    )
