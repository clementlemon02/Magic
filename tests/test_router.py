"""Router classification and the guards around it."""

from src.agents.router import PROMPT_TEMPLATE, _parse, classify, route_node
from src.agents.router_examples import EXAMPLES
from src.config import get_settings
from src.graph.state import UserContext


def _user() -> UserContext:
    return UserContext(id=1, role="support", dept="support", clearance_level=0)


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


def test_clarifies_on_an_ambiguous_question():
    state = {"query": "what about the other one?", "clarification_question": None}
    assert route_node(state, chat_model=FakeChat("clarify"))["route"] == "clarify"


# --- the fast path may not stall a request -------------------------------------

def test_the_student_is_not_allowed_to_choose_clarify(monkeypatch):
    """A wrong rag or sql still passes the ACL filter and the Verifier, so the worst
    case is a refusal. A wrong `clarify` stalls a real question — and the student,
    trained on 333 ordinary ones, reads a long rambling question as a pointer."""
    from src.agents import router

    monkeypatch.setenv("ROUTER_STUDENT_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(router, "student_probabilities",
                        lambda q, e=None: {"clarify": 0.99, "rag": 0.01, "sql": 0.0})
    asked = []

    class Teacher:
        def invoke(self, prompt):
            asked.append(prompt)
            return type("Reply", (), {"content": "rag"})()

    out = router.route_node({"query": "a long rambling question", "user": _user()},
                            chat_model=Teacher())
    assert out["route"] == "rag", "clarify must go to the teacher however sure the student is"
    assert asked, "the teacher was never consulted"
    get_settings.cache_clear()


def test_the_student_still_decides_rag_and_sql_on_its_own(monkeypatch):
    from src.agents import router

    monkeypatch.setenv("ROUTER_STUDENT_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(router, "student_probabilities",
                        lambda q, e=None: {"sql": 0.99, "rag": 0.01, "clarify": 0.0})

    class Teacher:
        def invoke(self, prompt):
            raise AssertionError("the teacher should not have been called")

    out = router.route_node({"query": "how many in August?", "user": _user()},
                            chat_model=Teacher())
    assert out["route"] == "sql"
    get_settings.cache_clear()


def test_decline_is_selectable():
    assert classify("hi", chat_model=FakeChat("decline")) == "decline"


def test_the_prompt_forbids_declining_a_hostile_question():
    """The §1 line. `decline` is for the KIND of message — greetings, general
    knowledge, "write me a poem". A hostile or probing question about company
    material is still a question about company material: it routes `rag`, meets the
    ACL predicate and the Verifier, and comes back as the generic refusal.

    Widening `decline` to cover "obviously adversarial" input would move access
    control into this prompt, which CLAUDE.md §1 forbids outright, and would swap a
    structural guarantee for a model's judgement about intent. Measured end to end
    in evals/adversarial_probe.py; pinned here so the prompt cannot drift quietly.
    """
    prompt = PROMPT_TEMPLATE.lower()
    assert "do not decline them" in prompt
    assert "aggressive" in prompt and "may not be allowed to see" in prompt

    # And the few-shot has to show it, not just say it: an injection labelled `rag`.
    hostile = [e for e in EXAMPLES if "ignore your instructions" in e["query"].lower()]
    assert hostile and all(e["route"] == "rag" for e in hostile)


def test_a_model_inventing_a_route_still_falls_back_to_rag():
    """Unchanged by adding `decline`: the fallback must stay `rag`, never the new
    route. Defaulting to `decline` would refuse real questions on a bad parse."""
    assert _parse("outofscope") == "rag"
    assert _parse("") == "rag"
