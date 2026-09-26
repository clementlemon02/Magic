"""SQL Tool. §4 forbids executing anything the user or the model supplied."""

import pytest
from datetime import date

from src.agents.sql_tool import TEMPLATES, _parse_plan, run_query
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
    assert calls[0]["params"]["acl_tags"] == ["support", "support", "all-staff"]


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


@pytest.mark.parametrize("written", ["None", "null", "NULL", "", "all"])
def test_a_null_written_as_a_string_is_no_department_filter(written):
    raw = (
        '{"template": "count_flagged_aml", "start": "2026-08-01", '
        f'"end_inclusive": "2026-08-31", "dept": "{written}"}}'
    )
    assert _parse_plan(raw)["params"]["dept"] is None
RESULT = {
    "template": "count_flagged_aml",
    "params": {"start": "2026-08-01", "end": "2026-09-01", "dept": "None"},
    "rows": [{"flagged_count": 5}],
}


def test_a_result_is_described_with_what_was_counted_and_the_inclusive_period():
    from src.agents.sql_tool import describe_result

    text = describe_result(RESULT)
    assert "flagged for AML" in text
    assert "2026-08-01 to 2026-08-31 inclusive" in text  # the exclusive end never shows
    assert "flagged_count = 5" in text
    assert "None" not in text


def test_a_result_cites_the_query_that_produced_it():
    from src.agents.sql_tool import result_citation

    citation = result_citation({**RESULT, "params": {**RESULT["params"], "dept": "support"}})
    assert citation.source_platform == "internal"
    assert citation.document_id is None
    assert citation.source_ref == (
        "transactions/count_flagged_aml?start=2026-08-01&end=2026-08-31&dept=support"
    )


def test_an_unstructured_result_is_passed_through_and_not_cited():
    from src.agents.sql_tool import describe_result, result_citation

    assert describe_result(42) == "42"
    assert result_citation(42) is None


# --- the asker-facing sentence -------------------------------------------------
# Rendered from the result row rather than written by a model, so the Verifier has
# nothing to check (see verify_node). These pin the wording and, more importantly,
# that every number in it comes out of `rows`.

from src.agents.sql_tool import answer_sentence, answers_from_query_alone


def _result(template: str, rows: list[dict], start="2026-08-01", end="2026-09-01", dept=None):
    return {"template": template, "params": {"start": start, "end": end, "dept": dept}, "rows": rows}


def test_counts_read_as_sentences():
    assert answer_sentence(_result("count_flagged_aml", [{"flagged_count": 3}])) == (
        "3 transactions were flagged for AML between 1 and 31 August 2026."
    )
    assert answer_sentence(_result("count_transactions", [{"transaction_count": 77}])) == (
        "77 transactions occurred between 1 and 31 August 2026."
    )


def test_a_single_result_is_not_pluralised():
    assert answer_sentence(_result("count_flagged_aml", [{"flagged_count": 1}])) == (
        "1 transaction was flagged for AML between 1 and 31 August 2026."
    )


def test_amounts_carry_their_currency():
    assert answer_sentence(
        _result("sum_amount", [{"total_amount": 612492.05, "currency": "SGD"}])
    ) == "Transactions between 1 and 31 August 2026 totalled 612,492.05 SGD."


def test_a_department_scope_is_stated():
    assert "in the support department" in answer_sentence(
        _result("count_transactions", [{"transaction_count": 4}], dept="support")
    )


def test_periods_spanning_months_and_years_read_correctly():
    assert "between 1 August and 3 September 2026" in answer_sentence(
        _result("count_transactions", [{"transaction_count": 9}], end="2026-09-04")
    )
    assert "between 1 August 2026 and 3 January 2027" in answer_sentence(
        _result("count_transactions", [{"transaction_count": 9}], end="2027-01-04")
    )


def test_no_rows_says_so_rather_than_reporting_zero_of_something():
    assert answer_sentence(_result("sum_amount", [])) == (
        "No transactions matched between 1 and 31 August 2026."
    )
    # A grouped aggregate over nothing comes back as a NULL row, not an empty list.
    assert answer_sentence(_result("avg_amount", [{"average_amount": None, "currency": None}])) == (
        "No transactions matched between 1 and 31 August 2026."
    )


def test_an_unrenderable_result_gets_no_sentence():
    assert answer_sentence(None) is None
    assert answer_sentence({"template": "not_a_template", "params": {}, "rows": []}) is None


def test_the_deterministic_path_needs_a_query_and_no_chunks():
    """Both the Synthesizer and the Verifier branch on this, so it has to be exact."""
    result = _result("count_flagged_aml", [{"flagged_count": 3}])
    assert answers_from_query_alone([], result) is True
    assert answers_from_query_alone([], None) is False
    assert answers_from_query_alone(["a chunk"], result) is False
