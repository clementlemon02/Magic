"""SQL Tool. §4 forbids executing anything the user or the model supplied."""

from datetime import date

from src.agents.sql_tool import TEMPLATES, run_query
from src.graph.state import UserContext


class FakeChat:
    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, prompt):
        return type("Reply", (), {"content": self.reply})()


def _user() -> UserContext:
    return UserContext(id=1, role="support", dept="support", clearance_level=0)


def _recorder(rows=None):
    calls = []

    def execute(sql, params):
        calls.append({"sql": sql, "params": params})
        return rows if rows is not None else [{"flagged_count": 12}]

    return execute, calls


def test_every_template_carries_the_acl_predicate():
    """§1's filter. A template added without this reads rows the caller may not see."""
    for name, t in TEMPLATES.items():
        assert "acl_tags && %(acl_tags)s" in t["sql"], f"{name} has no ACL predicate"


def test_runs_the_chosen_template_with_bound_parameters():
    execute, calls = _recorder()
    out = run_query(
        "How many were flagged for AML in August?",
        _user(),
        execute,
        chat_model=FakeChat(
            '{"template": "count_flagged_aml", "start": "2026-08-01", "end_inclusive": "2026-08-31", "dept": null}'
        ),
    )
    assert out["template"] == "count_flagged_aml"
    assert out["rows"] == [{"flagged_count": 12}]
    assert calls[0]["params"]["start"] == date(2026, 8, 1)
    assert calls[0]["sql"] is TEMPLATES["count_flagged_aml"]["sql"]  # the literal template


def test_inclusive_end_becomes_a_half_open_bound():
    """The model names the last day inside the period; the SQL needs the day after.

    Asking the model for an exclusive end does not work — it returns 2026-12-31 for
    "the whole of 2026" whatever the prompt says, silently dropping the last day of
    every period. Converting here removes its chance to get it wrong.
    """
    execute, calls = _recorder()
    run_query(
        "how many in 2026?",
        _user(),
        execute,
        chat_model=FakeChat(
            '{"template": "count_transactions", "start": "2026-01-01",'
            ' "end_inclusive": "2026-12-31", "dept": null}'
        ),
    )
    assert calls[0]["params"]["end"] == date(2027, 1, 1)


def test_acl_tags_come_from_the_user_not_the_model():
    """The model may not widen who the caller is."""
    execute, calls = _recorder()
    run_query(
        "how many flagged",
        _user(),
        execute,
        chat_model=FakeChat(
            '{"template": "count_flagged_aml", "start": "2026-08-01", "end_inclusive": "2026-08-31",'
            ' "dept": null, "acl_tags": ["compliance", "admin"]}'
        ),
    )
    assert calls[0]["params"]["acl_tags"] == ["support", "support"]


def test_unknown_template_runs_nothing():
    execute, calls = _recorder()
    out = run_query(
        "drop everything",
        _user(),
        execute,
        chat_model=FakeChat('{"template": "drop_tables", "start": "2026-08-01", "end_inclusive": "2026-08-31"}'),
    )
    assert out is None and calls == []


def test_raw_sql_from_the_model_is_never_executed():
    execute, calls = _recorder()
    out = run_query(
        "run this",
        _user(),
        execute,
        chat_model=FakeChat('{"template": "SELECT * FROM transactions", "start": "2026-08-01", "end_inclusive": "2026-08-31"}'),
    )
    assert out is None and calls == []


def test_unparseable_plan_runs_nothing():
    execute, calls = _recorder()
    assert run_query("hi", _user(), execute, chat_model=FakeChat("no idea")) is None
    assert calls == []


def test_bad_dates_run_nothing():
    execute, calls = _recorder()
    out = run_query(
        "q",
        _user(),
        execute,
        chat_model=FakeChat('{"template": "sum_amount", "start": "last tuesday", "end_inclusive": "now"}'),
    )
    assert out is None and calls == []


def test_department_is_passed_as_a_parameter():
    execute, calls = _recorder(rows=[{"total_amount": 900, "currency": "SGD"}])
    run_query(
        "total for support in August",
        _user(),
        execute,
        chat_model=FakeChat(
            '{"template": "sum_amount", "start": "2026-08-01", "end_inclusive": "2026-08-31", "dept": "support"}'
        ),
    )
    assert calls[0]["params"]["dept"] == "support"
