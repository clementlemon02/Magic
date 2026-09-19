"""Router classification and the guards around it."""

from src.agents.router import _parse, classify, route_node


class FakeChat:
    """Stands in for ChatHunyuan so these run without credentials."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return type("Reply", (), {"content": self.reply})()


def test_classifies_a_clean_one_word_answer():
    assert classify("How many payments were flagged?", chat_model=FakeChat("sql")) == "sql"


def test_recovers_a_route_from_a_chatty_answer():
    assert classify("anything", chat_model=FakeChat("The answer is: sql.")) == "sql"


def test_unparseable_answer_falls_back_to_rag():
    """rag is safe — its chunks are still ACL-filtered in SQL (§1), so a misroute cannot leak."""
    assert classify("anything", chat_model=FakeChat("I'm not sure what you mean")) == "rag"


def test_parse_ignores_case_and_punctuation():
    assert _parse("  Clarify.  ") == "clarify"


def test_prompt_carries_the_few_shot_examples():
    chat = FakeChat("rag")
    classify("What is the refund window?", chat_model=chat)
    assert "How many transactions were flagged for AML last month?" in chat.prompts[0]


def test_clarifies_only_once():
    """§4: one clarification round. A second would bounce the user forever."""
    state = {"query": "what about the other one?", "clarification_question": "Which ticket?"}
    assert route_node(state, chat_model=FakeChat("clarify"))["route"] == "rag"


def test_clarifies_on_the_first_ambiguous_question():
    state = {"query": "what about the other one?", "clarification_question": None}
    assert route_node(state, chat_model=FakeChat("clarify"))["route"] == "clarify"
