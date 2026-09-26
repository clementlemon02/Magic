"""Integration tests against a live Postgres (docker compose up -d).

Skipped when no database is reachable, so the suite stays green without Docker.
These cover what unit tests with a fake executor cannot: that the SQL templates
are valid Postgres, and that the §1 ACL predicate actually excludes rows.
"""

from datetime import date, datetime

import pytest

from src.agents.sql_tool import TEMPLATES, _psycopg_execute, run_query
from src.api.main import load_user
from src.config import get_settings
from src.graph.state import UserContext


def _reachable() -> bool:
    try:
        import psycopg

        with psycopg.connect(get_settings().database_url, connect_timeout=3):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="no database; run docker compose up -d")


class FakeChat:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, prompt):
        return type("Reply", (), {"content": self.reply})()


SUPPORT = UserContext(id=1, role="support", dept="support", clearance_level=0)


@pytest.fixture
def seeded_transactions():
    """Two support rows and one compliance-only row, committed then removed.

    Removes only its own rows, by id. It used to DELETE every transaction, which
    silently wiped the demo data whenever the suite ran against a seeded database.
    """
    import psycopg

    rows = [
        ("support", 100.00, False, datetime(2031, 8, 5), ["support"]),
        ("support", 250.00, True, datetime(2031, 8, 12), ["support"]),
        ("compliance", 999999.00, True, datetime(2031, 8, 20), ["compliance"]),
    ]
    with psycopg.connect(get_settings().database_url) as conn:
        ids = [
            conn.execute(
                """INSERT INTO transactions
                   (account_dept, amount, flagged_aml, occurred_at, acl_tags)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                row,
            ).fetchone()[0]
            for row in rows
        ]
        conn.commit()
    yield
    with psycopg.connect(get_settings().database_url) as conn:
        conn.execute("DELETE FROM transactions WHERE id = ANY(%s)", (ids,))
        conn.commit()


def test_load_user_resolves_a_seeded_persona():
    alex = load_user(1)
    assert alex is not None
    assert (alex.role, alex.dept, alex.clearance_level) == ("support", "support", 0)


def test_load_user_returns_none_for_an_unknown_id():
    assert load_user(99999) is None


def test_marcus_has_higher_clearance_than_alex():
    assert load_user(2).clearance_level > load_user(1).clearance_level


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_every_template_is_valid_postgres(name, seeded_transactions):
    """A fake executor cannot catch a syntax error. This runs each template for real."""
    _psycopg_execute(
        TEMPLATES[name]["sql"],
        {
            "start": date(2031, 8, 1),
            "end": date(2031, 9, 1),
            "dept": None,
            "acl_tags": SUPPORT.acl_tags(),
        },
    )


def test_acl_predicate_excludes_rows_the_caller_cannot_see(seeded_transactions):
    """§1 against a real database: the compliance-only row must not be counted."""
    out = run_query(
        "how many were flagged for AML in August?",
        SUPPORT,
        _psycopg_execute,
        chat_model=FakeChat(
            '{"template": "count_flagged_aml", "start": "2031-08-01", "end_inclusive": "2031-08-31", "dept": null}'
        ),
    )
    # Two flagged rows exist in range; only one carries a tag Alex holds.
    assert out["rows"][0]["flagged_count"] == 1


def test_a_compliance_caller_sees_the_restricted_row(seeded_transactions):
    """The same data, a different caller — the predicate is the only difference.

    Asserts on the total, not a count: each caller can see exactly one flagged
    row, so a count of 1 cannot tell which row the predicate matched.
    """
    marcus = UserContext(id=2, role="compliance", dept="compliance", clearance_level=1)
    out = run_query(
        "total value in August",
        marcus,
        _psycopg_execute,
        chat_model=FakeChat(
            '{"template": "sum_amount", "start": "2031-08-01", "end_inclusive": "2031-08-31", "dept": null}'
        ),
    )
    total = sum(r["total_amount"] for r in out["rows"])
    # The restricted row and nothing else: clearance 1 does not also grant the
    # support rows Alex sees. Overlapping tags decide this, never clearance.
    assert total == 999999


def test_sum_amount_excludes_the_restricted_row(seeded_transactions):
    out = run_query(
        "total value in August",
        SUPPORT,
        _psycopg_execute,
        chat_model=FakeChat(
            '{"template": "sum_amount", "start": "2031-08-01", "end_inclusive": "2031-08-31", "dept": null}'
        ),
    )
    total = sum(r["total_amount"] for r in out["rows"])
    assert total == 350  # 100 + 250, never the 999999 compliance row
