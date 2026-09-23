"""Clarification. One round only, so the question has to be usable."""

from src.agents.clarification import FALLBACK_QUESTION, clarify


class FakeChat:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, prompt):
        return type("Reply", (), {"content": self.reply})()


def test_returns_the_models_question():
    assert clarify("that thing", chat_model=FakeChat("Which ticket do you mean?")) == (
        "Which ticket do you mean?"
    )


def test_empty_reply_falls_back():
    assert clarify("that thing", chat_model=FakeChat("   ")) == FALLBACK_QUESTION


def test_rambling_reply_falls_back():
    assert clarify("that thing", chat_model=FakeChat("word " * 200)) == FALLBACK_QUESTION
