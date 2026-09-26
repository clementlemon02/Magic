"""Verifier parsing. Every unreadable judgement must fail CLOSED."""

import math

import pytest

from src.agents.verifier import _grounded_probability, _parse, _verify_measured, verify
from src.graph.state import Chunk, Citation


class FakeChat:
    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, prompt):
        return type("Reply", (), {"content": self.reply})()


def _chunk() -> Chunk:
    return Chunk(
        id=1,
        document_id=1,
        content="Chargebacks must be contested within 45 days.",
        acl_tags=["support"],
        score=0.9,
        citation=Citation(
            document_id=1, title="Refund Policy", source_platform="confluence", source_ref="S/1"
        ),
    )


def test_clean_json_parses():
    r = _parse('{"grounded": true, "unsupported": [], "confidence": 0.9}')
    assert r.grounded is True and r.confidence == 0.9


def test_code_fenced_json_parses():
    r = _parse('```json\n{"grounded": true, "unsupported": [], "confidence": 0.8}\n```')
    assert r.grounded is True


def test_json_wrapped_in_prose_parses():
    r = _parse('Sure! {"grounded": false, "unsupported": ["x"], "confidence": 0.7} Hope that helps.')
    assert r.grounded is False and r.unsupported == ["x"]


def test_unparseable_fails_closed():
    r = _parse("I think the answer looks fine to me")
    assert r.grounded is False
    assert r.confidence == 0.0  # below any threshold, so it escalates


def test_percentage_confidence_is_normalised():
    """Judges asked for 0.0-1.0 sometimes answer 95. That reading is unambiguous."""
    r = _parse('{"grounded": true, "unsupported": [], "confidence": 95}')
    assert r.confidence == 0.95


def test_nonsense_confidence_fails_closed():
    assert _parse('{"grounded": true, "unsupported": [], "confidence": 500}').confidence == 0.0
    assert _parse('{"grounded": true, "unsupported": [], "confidence": "high"}').confidence == 0.0


def test_named_unsupported_claims_override_a_grounded_flag():
    """A judge that lists failures and still says grounded is reporting a failure."""
    r = _parse('{"grounded": true, "unsupported": ["the 45 day figure"], "confidence": 0.9}')
    assert r.grounded is False


def test_missing_draft_is_ungrounded_without_calling_the_model():
    r = verify("q", None, [_chunk()], chat_model=None)
    assert r.grounded is False
    assert "no answer was drafted" in r.unsupported[0]


def test_verify_passes_evidence_into_the_prompt():
    seen = {}

    class Recorder(FakeChat):
        def invoke(self, prompt):
            seen["prompt"] = prompt
            return super().invoke(prompt)

    verify(
        "How long?",
        "45 days.",
        [_chunk()],
        chat_model=Recorder('{"grounded": true, "unsupported": [], "confidence": 0.9}'),
    )
    assert "contested within 45 days" in seen["prompt"]


# --- measured confidence -------------------------------------------------------
# `confidence` in the reply is a number the model was asked to invent. These cover
# reading the real one off the token it emitted for "grounded".


def _tok(token: str, logprob: float, tops: dict[str, float] | None = None) -> dict:
    return {
        "token": token,
        "logprob": logprob,
        "top_logprobs": [{"token": t, "logprob": lp} for t, lp in (tops or {}).items()],
    }


def _reply_tokens(verdict: str, tops: dict[str, float]) -> list[dict]:
    """The token stream for `{"grounded": <verdict>, ...}`."""
    return [
        _tok('{"', -0.01),
        _tok("grounded", -0.01),
        _tok('":', -0.01),
        _tok(f" {verdict}", tops[verdict], tops),
        _tok(",", -0.01),
    ]


def test_probability_is_renormalised_over_true_and_false():
    # exp(-0.1) / (exp(-0.1) + exp(-2.3)) — the rest of the vocabulary is irrelevant.
    tokens = _reply_tokens("true", {"true": -0.1, "false": -2.3})
    expected = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.3))
    assert _grounded_probability(tokens) == pytest.approx(expected)


def test_a_hedged_verdict_scores_near_a_coin_flip():
    """The point of the whole exercise: an uncertain judge now reports as uncertain."""
    tokens = _reply_tokens("false", {"true": -0.69, "false": -0.70})
    assert 0.45 < _grounded_probability(tokens) < 0.55


def test_true_before_the_grounded_key_is_ignored():
    """A stray `true` in preamble must not be mistaken for the verdict."""
    tokens = [_tok("true", -0.5, {"true": -0.5})] + _reply_tokens(
        "false", {"true": -4.0, "false": -0.02}
    )
    assert _grounded_probability(tokens) > 0.9  # the false token's own certainty


def test_emitted_token_counts_even_when_top_logprobs_omits_it():
    tokens = _reply_tokens("true", {"true": -0.05})
    assert _grounded_probability(tokens) == pytest.approx(1.0)


def test_no_verdict_token_measures_nothing():
    assert _grounded_probability([_tok("hello", -0.1)]) is None
    assert _grounded_probability([]) is None


def test_measured_confidence_replaces_the_self_reported_one(monkeypatch):
    monkeypatch.setattr(
        "src.llm.factory.chat_with_logprobs",
        lambda prompt: (
            '{"grounded": true, "unsupported": [], "confidence": 1.0}',
            _reply_tokens("true", {"true": -0.69, "false": -0.70}),
        ),
    )
    result = _verify_measured("prompt")
    assert result.grounded is True
    assert result.confidence < 0.55  # not the 1.0 the model claimed


def test_an_unreadable_judgement_stays_at_zero_confidence(monkeypatch):
    """Fail closed: certainty about a malformed verdict is still no verdict."""
    monkeypatch.setattr(
        "src.llm.factory.chat_with_logprobs",
        lambda prompt: ("not json at all", _reply_tokens("true", {"true": -0.001})),
    )
    result = _verify_measured("prompt")
    assert result.grounded is False and result.confidence == 0.0


def test_falls_back_when_the_backend_cannot_report_logprobs(monkeypatch):
    monkeypatch.setattr("src.llm.factory.chat_with_logprobs", lambda prompt: None)
    assert _verify_measured("prompt") is None


def test_spellings_of_one_verdict_add_up():
    """Regression: ' true', 'true' and ' True' are one answer, not three candidates.

    Keying them into a dict and assigning kept the LAST — the least likely spelling —
    which inverted the measurement: a confident `true` scored 0.002.
    """
    verdict = _tok(" true", -0.0)
    verdict["top_logprobs"] = [
        {"token": " true", "logprob": -0.0},
        {"token": "true", "logprob": -10.2},
        {"token": " false", "logprob": -11.6},
        {"token": " True", "logprob": -17.8},
    ]
    tokens = [_tok('{"', -0.01), _tok("ground", -0.0), _tok("ed", -0.0), _tok('":', -0.0), verdict]
    assert _grounded_probability(tokens) > 0.99
