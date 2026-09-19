"""Synthesizer. Drafts only from evidence the caller was permitted to see."""

from src.agents.synthesizer import synthesize
from src.graph.state import Chunk, Citation


class FakeChat:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        self.last = prompt
        return type("Reply", (), {"content": self.reply})()


def _chunk(doc_id: int, content: str) -> Chunk:
    return Chunk(
        id=doc_id * 10,
        document_id=doc_id,
        content=content,
        acl_tags=["support"],
        score=0.9,
        citation=Citation(
            document_id=doc_id,
            title=f"Doc {doc_id}",
            source_platform="confluence",
            source_ref=f"S/{doc_id}",
        ),
    )


def test_drafts_from_chunks_and_cites_them():
    draft, citations = synthesize(
        "How long?", [_chunk(1, "45 days.")], chat_model=FakeChat("45 days.")
    )
    assert draft == "45 days."
    assert [c.document_id for c in citations] == [1]


def test_citations_are_deduplicated_per_document():
    chunks = [_chunk(1, "a"), _chunk(1, "b"), _chunk(2, "c")]
    _, citations = synthesize("q", chunks, chat_model=FakeChat("answer"))
    assert [c.document_id for c in citations] == [1, 2]


def test_insufficient_evidence_drafts_nothing():
    """The Verifier then reports ungrounded, so the hop loop or escalation decides."""
    draft, citations = synthesize("q", [_chunk(1, "unrelated")], chat_model=FakeChat("INSUFFICIENT"))
    assert draft is None and citations == []


def test_no_permitted_evidence_skips_the_model_entirely():
    chat = FakeChat("should never run")
    draft, citations = synthesize("q", [], sql_result=None, chat_model=chat)
    assert draft is None and citations == [] and chat.calls == 0


def test_sql_result_is_usable_evidence_on_its_own():
    chat = FakeChat("128 transactions.")
    draft, citations = synthesize("how many?", [], sql_result={"rows": [{"c": 128}]}, chat_model=chat)
    assert draft == "128 transactions."
    assert citations == []  # a computed figure has no document to cite
    assert "128" in chat.last
