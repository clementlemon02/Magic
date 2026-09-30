"""Verifier parsing. Every unreadable judgement must fail CLOSED."""

import math

import pytest

from src.agents.verifier import (
    _grounded_in_figures,
    _grounded_probability,
    _parse,
    _verify_measured,
    unsupported_quantities,
    verify,
)
from src.graph.state import Chunk, Citation, VerificationResult


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


# --- a query answer has nothing to verify --------------------------------------

def test_a_query_only_answer_skips_the_judge(monkeypatch):
    """The Synthesizer rendered it from the row, so there is no invented claim to find.

    Not a shortcut around §4: the judge exists to catch a model inventing something,
    and nothing here was written by a model.
    """
    from src.agents import verifier

    def fail(*args, **kwargs):
        raise AssertionError("the verifier must not call a model for a query answer")

    monkeypatch.setattr(verifier, "_verify_measured", fail)
    out = verifier.verify_node({
        "query": "how many flagged?",
        "draft_answer": "3 transactions were flagged for AML between 1 and 31 August 2026.",
        "retrieved_chunks": [],
        "sql_result": {"template": "count_flagged_aml",
                       "params": {"start": "2026-08-01", "end": "2026-09-01", "dept": None},
                       "rows": [{"flagged_count": 3}]},
        "citations": [],
    })
    result = out["verification"]
    assert result.grounded is True and result.confidence == 1.0
    # Confidence 1.0 on purpose: VERIFIER_CONFIDENCE_THRESHOLD must not escalate a
    # number that came straight out of the database.
    assert result.unsupported == []


def test_a_rag_answer_still_goes_to_the_judge():
    """The skip is for query-only answers; evidence-backed drafts are judged as before."""
    from src.agents.verifier import verify_node

    chunk = _chunk()
    out = verify_node(
        {
            "query": "how long to contest?",
            "draft_answer": "Customers have 45 days.",
            "retrieved_chunks": [chunk],
            "sql_result": None,
            "citations": [chunk.citation],
        },
        chat_model=FakeChat('{"grounded": true, "unsupported": [], "confidence": 0.9}'),
    )
    assert out["verification"].confidence == 0.9  # the judge's number, not 1.0


# --- the figure check ----------------------------------------------------------

def _ev(text: str) -> list[Chunk]:
    return [Chunk(id=1, document_id=1, content=text, acl_tags=["support"], score=0.9,
                  citation=Citation(document_id=1, title="t", source_platform="slack", source_ref="r"))]


def test_a_figure_the_evidence_never_states_is_unsupported():
    """The real fail-open case. The passage is about a backlog being cleared and names
    no threshold at all, yet qwen2.5:7b judged this answer grounded at 0.924."""
    missing = unsupported_quantities(
        "No, refunds above SGD 2,000 need team lead approval before they are issued.",
        "Can I approve a SGD 3,000 refund myself?",
        _ev("the refund backlog is fully cleared, and requests are reviewed within five business days"),
    )
    assert missing == [2000.0]


def test_the_question_counts_as_a_source():
    """An asker's own figure may be repeated back. Otherwise no question containing a
    number could ever be answered."""
    assert unsupported_quantities(
        "A SGD 3,000 refund needs approval.", "Can I approve a SGD 3,000 refund myself?",
        _ev("refunds need approval"),
    ) == []


def test_separators_and_spelled_numbers_do_not_cause_a_false_refusal():
    assert unsupported_quantities("Up to SGD 2,000.", "q", _ev("the limit is 2000 dollars")) == []
    assert unsupported_quantities("Reviewed within 5 days.", "q", _ev("within five business days")) == []


def test_the_check_only_ever_downgrades():
    """One direction. An ungrounded verdict is left exactly as the judge left it, and
    a grounded one is never manufactured."""
    judged_bad = VerificationResult(grounded=False, unsupported=["made up"], confidence=0.4)
    assert _grounded_in_figures(judged_bad, "45 days", "q", _ev("45 days"), None) is judged_bad

    judged_good = VerificationResult(grounded=True, unsupported=[], confidence=0.92)
    out = _grounded_in_figures(judged_good, "SGD 2,000", "q", _ev("no figures here"), None)
    assert out.grounded is False
    assert "2,000" in out.unsupported[-1]
    # Certain, not unsure: the figure is absent, and a low confidence would read to
    # VERIFIER_CONFIDENCE_THRESHOLD as doubt about the verdict rather than about the answer.
    assert out.confidence == 1.0


def test_a_grounded_answer_whose_figures_are_all_present_is_left_alone():
    judged = VerificationResult(grounded=True, unsupported=[], confidence=0.88)
    assert _grounded_in_figures(
        judged, "Customers have 45 days.", "How long?", _ev("contest within 45 days"), None
    ) is judged


# --- the gated multi-passage recheck --------------------------------------------

class _RoutingChat:
    """Replies based on which chunk's content is in the prompt, and counts calls —
    so a test can assert exactly how many extra model calls a fix costs."""

    def __init__(self, replies: dict[str, str], default: str):
        self.replies = replies
        self.default = default
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        for marker, reply in self.replies.items():
            if marker in prompt:
                return type("Reply", (), {"content": reply})()
        return type("Reply", (), {"content": self.default})()


def _labelled_chunk(n: int, content: str) -> Chunk:
    return Chunk(
        id=n, document_id=n, content=content, acl_tags=["support"], score=0.9,
        citation=Citation(document_id=n, title="t", source_platform="confluence", source_ref=f"r{n}"),
    )


UNGROUNDED = '{"grounded": false, "unsupported": ["x"], "confidence": 0.8}'
GROUNDED = '{"grounded": true, "unsupported": [], "confidence": 0.95}'


def test_allow_recheck_defaults_off():
    """The property that matters most: omitting the flag must cost exactly one model
    call, whatever the evidence looks like — a live request cannot afford more."""
    chat = _RoutingChat({}, default=UNGROUNDED)  # every prompt refuses
    chunks = [_labelled_chunk(1, "clearly states the rule"), _labelled_chunk(2, "unrelated")]
    result = verify("q", "an answer", chunks, chat_model=chat)
    assert not result.grounded
    assert chat.calls == 1, "the recheck fired despite allow_recheck being omitted"


def test_allow_recheck_true_rescues_a_passage_the_joint_read_missed():
    """The fix, opted into. One passage alone grounds it even though the joint
    prompt (which contains both chunks) does not."""
    chat = _RoutingChat({"THE-CLEAR-ONE": GROUNDED}, default=UNGROUNDED)
    chunks = [_labelled_chunk(1, "THE-CLEAR-ONE states it plainly"), _labelled_chunk(2, "confusing filler")]
    result = verify("q", "an answer", chunks, chat_model=chat, allow_recheck=True)
    assert result.grounded
    # joint (1) + at most len(chunks) rechecks (2) — bounded, not unbounded
    assert chat.calls <= 3


def test_allow_recheck_true_still_refuses_when_nothing_alone_grounds_it():
    """Must not invent grounding no single passage gives — every call, joint and
    per-passage, genuinely refuses here."""
    chat = _RoutingChat({}, default=UNGROUNDED)
    chunks = [_labelled_chunk(1, "a"), _labelled_chunk(2, "b"), _labelled_chunk(3, "c")]
    result = verify("q", "an answer", chunks, chat_model=chat, allow_recheck=True)
    assert not result.grounded
    assert chat.calls == 1 + len(chunks), "should try every passage before giving up"


def test_recheck_never_fires_with_a_single_chunk():
    """Nothing to recheck against — nothing else the answer could have come from."""
    chat = _RoutingChat({}, default=UNGROUNDED)
    result = verify("q", "an answer", [_labelled_chunk(1, "only chunk")], chat_model=chat, allow_recheck=True)
    assert not result.grounded
    assert chat.calls == 1


def test_verify_node_never_turns_the_recheck_on():
    """A live request cannot afford it — measured worst case ~13s, and a full
    refusal_timing sweep with it unconditional took escapes from 0% to 5.7% against
    the <=1% target. This is the one place that must never pass allow_recheck=True;
    grep rather than trust a docstring, since a docstring does not fail CI."""
    import inspect

    from src.agents import verifier

    source = inspect.getsource(verifier.verify_node)
    assert "allow_recheck" not in source
