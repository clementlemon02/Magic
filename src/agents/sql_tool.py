"""SQL Tool — CLAUDE.md §4.

In: `query`, `user`. Out: `sql_result`.

Not text-to-SQL. §4 forbids executing anything the user supplied, so the model
only picks a template name and extracts parameters; the SQL itself is fixed here
and every value is bound by the driver. The ACL predicate is appended by this
module from the caller's UserContext, never by the model — that is the §1 filter,
and a template cannot opt out of it.
"""

import json
import re
from datetime import date, timedelta

from src.graph.state import Citation, GraphState, UserContext

# Every template carries `acl_tags && %(acl_tags)s`. A new template without it
# would read rows the caller may not see, so keep the predicate in the string.
TEMPLATES: dict[str, dict] = {
    "count_flagged_aml": {
        "describes": "how many transactions were flagged for AML in a period",
        "sql": """
            SELECT count(*) AS flagged_count
            FROM transactions
            WHERE flagged_aml
              AND occurred_at >= %(start)s AND occurred_at < %(end)s
              AND (%(dept)s::text IS NULL OR account_dept = %(dept)s)
              AND acl_tags && %(acl_tags)s
        """,
    },
    "count_transactions": {
        "describes": "how many transactions occurred in a period",
        "sql": """
            SELECT count(*) AS transaction_count
            FROM transactions
            WHERE occurred_at >= %(start)s AND occurred_at < %(end)s
              AND (%(dept)s::text IS NULL OR account_dept = %(dept)s)
              AND acl_tags && %(acl_tags)s
        """,
    },
    "sum_amount": {
        "describes": "the total value of transactions in a period",
        "sql": """
            SELECT coalesce(sum(amount), 0) AS total_amount, currency
            FROM transactions
            WHERE occurred_at >= %(start)s AND occurred_at < %(end)s
              AND (%(dept)s::text IS NULL OR account_dept = %(dept)s)
              AND acl_tags && %(acl_tags)s
            GROUP BY currency
        """,
    },
    "avg_amount": {
        "describes": "the average transaction value in a period",
        "sql": """
            SELECT round(avg(amount), 2) AS average_amount, currency
            FROM transactions
            WHERE occurred_at >= %(start)s AND occurred_at < %(end)s
              AND (%(dept)s::text IS NULL OR account_dept = %(dept)s)
              AND acl_tags && %(acl_tags)s
            GROUP BY currency
        """,
    },
}

PROMPT_TEMPLATE = """Pick the query that answers the question, and extract its parameters.

Available queries:
{catalogue}

Reply with JSON only, no prose and no code fences:
{{"template": "<name>", "start": "YYYY-MM-DD", "end_inclusive": "YYYY-MM-DD", "dept": "<department or null>"}}

"start" is the first day of the period and "end_inclusive" is the LAST DAY INSIDE it —
for August 2026 that is 2026-08-01 and 2026-08-31. Resolve relative periods against
today, {today}.
Quarters are calendar quarters: Q1 is Jan-Mar, Q2 Apr-Jun, Q3 Jul-Sep, Q4 Oct-Dec.
A named month means that whole month. "Last month" is the month before today's.
Set "dept" only if the question names a department; otherwise null.
If no query fits, reply {{"template": null}}.

Question: {query}

JSON:"""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _catalogue() -> str:
    return "\n".join(f"  {name} — {t['describes']}" for name, t in TEMPLATES.items())


def _parse_plan(raw: str) -> dict | None:
    text = _FENCE.sub("", str(raw)).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        plan = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    name = plan.get("template")
    if name not in TEMPLATES:
        return None  # includes an explicit null — no template fits

    # The model names the last day inside the period; the half-open bound the SQL
    # needs is computed here. Asking for an exclusive end directly does not work —
    # qwen2.5 returns 2026-12-31 for "the whole of 2026" whatever the prompt says,
    # which silently drops the last day of every period.
    try:
        params = {
            "start": date.fromisoformat(str(plan["start"])),
            "end": date.fromisoformat(str(plan["end_inclusive"])) + timedelta(days=1),
        }
    except (KeyError, TypeError, ValueError):
        return None

    # The model writes null as a string often enough to matter ("None", "null"), and
    # a filter on a department called "None" silently returns zero.
    dept = str(plan.get("dept") or "").strip()
    params["dept"] = None if dept.lower() in {"", "none", "null", "all", "any"} else dept
    return {"template": name, "params": params}


def run_query(
    query: str,
    user: UserContext,
    execute,
    chat_model=None,
    today: date | None = None,
) -> dict | None:
    """Plan and run one templated query. `execute(sql, params) -> list[dict]`."""
    if chat_model is None:
        from src.llm.factory import get_chat_model

        chat_model = get_chat_model()

    prompt = PROMPT_TEMPLATE.format(
        catalogue=_catalogue(), today=(today or date.today()).isoformat(), query=query
    )
    plan = _parse_plan(getattr(chat_model.invoke(prompt), "content", ""))
    if plan is None:
        return None

    # The caller's tags come from the verified UserContext, not the model's reply.
    params = {**plan["params"], "acl_tags": user.acl_tags()}
    rows = execute(TEMPLATES[plan["template"]]["sql"], params)

    return {
        "template": plan["template"],
        "params": {k: str(v) for k, v in plan["params"].items()},
        "rows": rows,
    }


def _period(result: dict) -> tuple[str, str, str | None]:
    p = result["params"]
    last_day = date.fromisoformat(p["end"]) - timedelta(days=1)
    dept = p.get("dept")
    return p["start"], last_day.isoformat(), None if dept in (None, "None") else dept


def _is_structured(result) -> bool:
    return isinstance(result, dict) and result.get("template") in TEMPLATES


def describe_result(result) -> str:
    """A query result as a sentence, for the Synthesizer and the Verifier.

    Handed the raw dict, the model echoed the bare value — the asker saw "5", with
    no hint of what was counted, over which period, or from where.
    """
    if not _is_structured(result):
        return str(result)
    start, last_day, dept = _period(result)
    scope = f"the {dept} department" if dept else "every department the asker may see"
    values = "; ".join(
        ", ".join(f"{k} = {v}" for k, v in row.items()) for row in result["rows"]
    ) or "no matching rows"
    what = TEMPLATES[result["template"]]["describes"]
    return f"Transactions query — {what}, from {start} to {last_day} inclusive, over {scope}: {values}."


def result_citation(result) -> Citation | None:
    """Cite the query itself, so a number on screen can be re-run and checked."""
    if not _is_structured(result):
        return None
    start, last_day, dept = _period(result)
    ref = f"transactions/{result['template']}?start={start}&end={last_day}"
    return Citation(
        document_id=None,
        title=f"Transactions: {TEMPLATES[result['template']]['describes']} ({start} to {last_day})",
        source_platform="internal",
        source_ref=ref + (f"&dept={dept}" if dept else ""),
    )


def _psycopg_execute(sql: str, params: dict) -> list[dict]:
    import psycopg
    from psycopg.rows import dict_row

    from src.config import get_settings

    # Read-only: a templated SELECT cannot write, and the transaction says so too.
    settings = get_settings()
    with psycopg.connect(
        settings.database_url,
        row_factory=dict_row,
        connect_timeout=settings.db_connect_timeout_seconds,
    ) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def sql_tool_node(state: GraphState, chat_model=None, execute=None) -> dict:
    return {
        "sql_result": run_query(
            state["query"],
            state["user"],
            execute or _psycopg_execute,
            chat_model=chat_model,
        )
    }
