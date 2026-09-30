"""Synthesizer. Drafts only from evidence the caller was permitted to see."""

from src.agents.synthesizer import _extractive_answer, _looks_compound, synthesize
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


def test_cites_only_the_document_the_answer_came_from():
    """No model cooperation needed — selection is from the answer's own wording."""
    chunks = [
        _chunk(1, "Chargebacks must be contested within 45 days of the transaction date."),
        _chunk(2, "Customer PII must be masked in all exported reports."),
        _chunk(3, "Payment outage ENG-4471 root cause was an expired TLS certificate."),
    ]
    _, citations = synthesize(
        "What caused the outage?",
        chunks,
        chat_model=FakeChat("The payment outage root cause was an expired TLS certificate."),
    )
    assert [c.document_id for c in citations] == [3]


def test_cites_both_documents_when_the_answer_spans_them():
    """The Scenario 1 shape: one answer, two platforms, both cited."""
    chunks = [
        _chunk(1, "Chargebacks must be contested within 45 days."),
        _chunk(2, "Customer PII must be masked in all exported reports."),
        _chunk(3, "On-call PII access goes through the standing approval process."),
    ]
    _, citations = synthesize(
        "How is PII handled and how does on-call get access?",
        chunks,
        chat_model=FakeChat(
            "Customer PII must be masked in exported reports, and on-call PII access "
            "goes through the standing approval process."
        ),
    )
    assert [c.document_id for c in citations] == [2, 3]


def test_paraphrased_answer_falls_back_to_citing_everything():
    """Below the overlap floor the signal is noise — broad beats wrong."""
    chunks = [_chunk(1, "Chargebacks must be contested within 45 days."), _chunk(2, "Unrelated.")]
    _, citations = synthesize("q", chunks, chat_model=FakeChat("Six weeks, roughly."))
    assert [c.document_id for c in citations] == [1, 2]


def test_empty_reply_is_treated_as_no_answer():
    """Returning "" would hand the asker a blank response."""
    draft, citations = synthesize("q", [_chunk(1, "a")], chat_model=FakeChat("   "))
    assert draft is None and citations == []


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


def test_a_sql_answer_is_rendered_from_the_row_and_cites_its_query():
    result = {
        "template": "count_flagged_aml",
        "params": {"start": "2026-08-01", "end": "2026-09-01", "dept": "None"},
        "rows": [{"flagged_count": 5}],
    }
    chat = FakeChat("a model must never be asked to restate a query result")
    draft, citations = synthesize("how many flagged?", [], sql_result=result, chat_model=chat)
    assert draft == "5 transactions were flagged for AML between 1 and 31 August 2026."
    assert [c.source_platform for c in citations] == ["internal"]
    # Rendered from the row, not drafted: restating it through a model would only add
    # a second chance to get the number wrong, and something for a judge to check.
    assert chat.calls == 0


# --- the extractive fast path ---------------------------------------------------
# Below the model-call layer: _extractive_answer never rewrites anything, so its
# safety case is "this exact sentence already sits in a permitted passage", not a
# judgement call. synthesize()'s own tests below cover it wired into the real path.

REFUND = _chunk(1, "Customers have 45 days from the transaction date to contest a "
                    "chargeback. Disputes raised after 45 days are declined unless a "
                    "manager approves an override.")


def test_extractive_answer_returns_the_leading_sentence_unedited():
    result = _extractive_answer("How long do customers have to contest a chargeback?", [REFUND], 0.25)
    assert result is not None
    sentence, source = result
    assert sentence == "Customers have 45 days from the transaction date to contest a chargeback."
    assert source.document_id == 1


def test_extractive_answer_requires_a_clear_lead():
    """A genuine tie has nothing to pick between — guessing would trade the fast
    path's whole safety case (this exact text is already known-permitted) for a
    coin flip, so it falls back to the model instead."""
    tied = [
        _chunk(1, "The payment outage affected the settlement gateway overnight."),
        _chunk(2, "The payment outage affected the settlement gateway overnight."),
    ]
    assert _extractive_answer("What caused the payment outage?", tied, 0.1) is None


def test_extractive_answer_requires_the_overlap_floor():
    unrelated = [_chunk(1, "The office coffee machine is on the third floor.")]
    assert _extractive_answer("How long to contest a chargeback?", unrelated, 0.25) is None


def test_extractive_answer_ignores_short_fragments():
    """A two-word fragment can win on Jaccard by chance alone — real prose clears
    this easily, so requiring a minimum length costs nothing on a genuine answer."""
    chunk = _chunk(1, "Approved. See the playbook for full details on timing and process.")
    assert _extractive_answer("Approved?", [chunk], 0.1) is None


def test_looks_compound_flags_coordinated_questions():
    assert _looks_compound("How is PII handled, and how does on-call get access?")
    assert _looks_compound("What is the policy on refunds or chargebacks?")
    assert not _looks_compound("How long do customers have to contest a chargeback?")


def test_extractive_answer_never_fires_on_a_compound_query():
    """The actual bug found this session: a two-part question can score AT OR ABOVE
    a genuine single-answer case depending only on phrasing, so no overlap floor
    separates them reliably — the query itself has to be checked instead."""
    chunks = [
        _chunk(2, "Customer PII must be masked in all exported reports."),
        _chunk(3, "On-call PII access goes through the standing approval process."),
    ]
    result = _extractive_answer(
        "How is PII handled and how does on-call get access?", chunks, 0.1
    )
    assert result is None


def test_synthesize_takes_the_extractive_path_and_never_calls_the_model():
    chat = FakeChat("should never run")
    draft, citations = synthesize(
        "How long do customers have to contest a chargeback?", [REFUND], chat_model=chat
    )
    assert draft == "Customers have 45 days from the transaction date to contest a chargeback."
    assert chat.calls == 0
    assert citations[0].document_id == 1
    assert citations[0].passage == REFUND.content  # so the asker can check it


def test_synthesize_falls_back_to_the_model_when_nothing_clearly_leads():
    chat = FakeChat("Customer PII is masked, and on-call access goes through approval.")
    chunks = [
        _chunk(2, "Customer PII must be masked in all exported reports."),
        _chunk(3, "On-call PII access goes through the standing approval process."),
    ]
    draft, _ = synthesize(
        "How is PII handled and how does on-call get access?", chunks, chat_model=chat
    )
    assert chat.calls == 1
    assert draft == chat.reply
