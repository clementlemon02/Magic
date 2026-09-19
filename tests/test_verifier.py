"""Verifier parsing. Every unreadable judgement must fail CLOSED."""

from src.agents.verifier import _parse, verify
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
