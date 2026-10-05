"""Brief scenario 2 end to end: a change at the source reaches answers, inside a bound.

    .venv/bin/python -m evals.freshness_probe --yes [API_URL]

Edits the mock sources (scripts/mock_source.py), asks the live API, syncs, asks again, and
puts everything back at the end. While it runs it changes what the server will answer, so point
it at a rehearsal instance and never at one somebody is presenting from; --yes says you did.

Start the server from THIS checkout: the mock sources are a file under it, and the probe and
the server find it by the same path. Needs Ollama, and QUERY_MAX_PER_WINDOW above ~30.

Every row is a claim that was untrue before src/ingestion/sync.py existed, except the last
few, which are claims the sync must not break: an officer's revoke, the answer cache, the audit
chain. Verdicts are asserted; the one thing only printed is what an asker sees BETWEEN an edit
and the next sync, because that window is the product's stated freshness bound, not a defect.
"""

import contextlib
import io
import json
import sys
import time
import urllib.error
import urllib.request

from scripts import mock_source
from src.graph.state import GENERIC_REFUSAL

API = "http://localhost:8000"
ALEX = ("alex.tan@aurelia.example", "demo")  # support, clearance 0
MARCUS = ("marcus.lim@aurelia.example", "demo")  # compliance officer
ALEX_ID = 1  # load-bearing (scripts/seed_users.sql)

REFUND = "confluence:SUPPORT/refund-policy"
OUTAGE = "jira:PAY/ENG-4471"
AML = "confluence:COMPLIANCE/aml-escalation"
Q_REFUND = "How long do customers have to contest a chargeback?"
Q_OUTAGE = "What caused the payment outage?"
Q_AML = "What triggers an AML escalation review?"


def _call(method: str, path: str, payload=None, token=None) -> tuple[int, dict | list]:
    request = urllib.request.Request(
        API + path, method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"content-type": "application/json",
                 **({"authorization": f"Bearer {token}"} if token else {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def _author(*args: str) -> None:
    """Change what a mock source says, as an author would, quietly."""
    with contextlib.redirect_stdout(io.StringIO()):
        mock_source.main(list(args))


def main(argv: list[str]) -> int:
    global API
    if "--yes" not in argv:
        print(__doc__)
        return 2
    API = next((a for a in argv if a.startswith("http")), API)

    try:
        alex = _call("POST", "/auth/login", {"email": ALEX[0], "password": ALEX[1]})[1]["access_token"]
        marcus = _call("POST", "/auth/login", {"email": MARCUS[0], "password": MARCUS[1]})[1]["access_token"]
    except (urllib.error.URLError, KeyError) as error:
        print(f"cannot sign in to {API}: {error}\nStart it: uvicorn src.api.main:app --port 8000")
        return 2

    rows: list[tuple[bool, str, str]] = []

    def claim(ok: bool, text: str, detail: str = "") -> None:
        rows.append((bool(ok), text, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {text}" + (f"   [{detail}]" if detail else ""))

    def ask(token: str, query: str) -> dict:
        return _call("POST", "/query", {"query": query}, token)[1]

    def cached(reply: dict) -> bool:
        return [s["node"] for s in reply.get("trace") or []] == ["cache"]

    def refused(reply: dict) -> bool:
        return reply["text"] == GENERIC_REFUSAL

    def cites(reply: dict, ref: str) -> bool:
        return any(c["source_ref"] == ref for c in reply.get("citations") or [])

    def sync() -> dict[str, dict]:
        started = time.monotonic()
        status, reports = _call("POST", "/admin/sync", None, marcus)
        assert status == 200, f"POST /admin/sync -> {status}: {reports}"
        print(f"        sync {time.monotonic() - started:4.1f}s  " + "  ".join(
            f"{r['platform']}:+{r['added']}~{r['updated']}-{r['removed']}"
            f"/g+{r['grants_added']}-{r['grants_revoked']}" for r in reports))
        errors = [r for r in reports if r["error"]]
        assert not errors, f"a source failed to sync: {errors}"
        return {r["platform"]: r for r in reports}

    def documents() -> dict[str, int]:
        return {s["platform"]: s["documents"] for s in _call("GET", "/admin/sources", None, marcus)[1]}

    print(f"freshness probe against {API}\n")
    _author("reset")
    sync()
    start = documents()
    try:
        print("1. An edit at the source")
        before = ask(alex, Q_REFUND)
        claim("45 days" in before["text"], "baseline: the policy says 45 days", before["text"][:60])
        claim(all(c.get("as_of") for c in before["citations"] if c["source_platform"] != "internal"),
              "every citation says when its source last synced",
              ", ".join(f"{c['source_ref']}@{(c.get('as_of') or '-')[11:19]}" for c in before["citations"]))
        _author("edit", REFUND, "45 days", "30 days")
        between = ask(alex, Q_REFUND)
        print(f"        between the edit and the sync an asker sees: {between['text'][:60]!r}"
              f"{'  (from the cache)' if cached(between) else ''}   <- the freshness window")
        report = sync()["confluence"]
        claim(report["updated"] == 1, "the sync found exactly the one edited document",
              f"updated={report['updated']}")
        after = ask(alex, Q_REFUND)
        claim("30 days" in after["text"] and "45 days" not in after["text"],
              "the next answer says 30 days", after["text"][:60])
        claim(not cached(after), "…and was not the cached pre-edit answer")

        print("2. Narrowing who may read it")
        _author("access", REFUND, "compliance")
        sync()
        reply = ask(alex, Q_REFUND)
        # Not "refused": drive:file-refund-playbook also answers this question and Alex may
        # still read it, so a correct answer can exist. What must not happen is the policy.
        claim(not cites(reply, "SUPPORT/refund-policy") and "30 days" not in reply["text"],
              "Alex, no longer entitled, is not answered from it",
              ", ".join(c["source_ref"] for c in reply.get("citations") or []) or "refused")
        reply = ask(marcus, Q_REFUND)
        claim(not refused(reply) and cites(reply, "SUPPORT/refund-policy"),
              "Marcus, still entitled, is answered with a citation")

        print("3. Restoring the source restores the answer")
        _author("restore", REFUND)
        sync()
        reply = ask(alex, Q_REFUND)
        claim("45 days" in reply["text"] and cites(reply, "SUPPORT/refund-policy"),
              "Alex is answered again, from the original text", reply["text"][:60])

        print("4. A deletion at the source")
        reply = ask(alex, Q_OUTAGE)
        claim(cites(reply, "PAY/ENG-4471"), "baseline: the outage answer cites ENG-4471")
        _author("delete", OUTAGE)
        sync()
        reply = ask(alex, Q_OUTAGE)
        claim(not cites(reply, "PAY/ENG-4471") and "tls" not in reply["text"].lower(),
              "the deleted issue is no longer cited or quoted", reply["text"][:60])
        _author("restore", OUTAGE)
        sync()
        claim(cites(ask(alex, Q_OUTAGE), "PAY/ENG-4471"), "…and returns when the source restores it")

        print("5. A RESTRICTED document narrowed at the source, with no sync")
        reply = ask(marcus, Q_AML)
        claim(not refused(reply) and cites(reply, "COMPLIANCE/aml-escalation"),
              "baseline: Marcus is answered from the AML procedure")
        _author("access", AML, "engineering")
        reply = ask(marcus, Q_AML)
        claim(refused(reply), "Marcus is refused AT ONCE, before any sync (the query-time recheck)",
              "served from the cache" if cached(reply) else "")
        sync()
        claim(refused(ask(marcus, Q_AML)), "…and still refused after the sync")
        _author("restore", AML)
        sync()
        claim(not refused(ask(marcus, Q_AML)), "…and answered again when the source restores it")

        print("6. What a sync must not undo or break")
        body = {"user_id": ALEX_ID, "source_platform": "confluence",
                "source_ref": "SUPPORT/refund-policy"}
        _call("POST", "/admin/permissions/revoke", body, marcus)
        sync()  # the source still backs the grant
        claim(not cites(ask(alex, Q_REFUND), "SUPPORT/refund-policy"),
              "an officer's revoke survives a sync that still backs it")
        _call("POST", "/admin/permissions/grant", body, marcus)
        claim(cites(ask(alex, Q_REFUND), "SUPPORT/refund-policy"), "…and the officer can grant it back")

        ask(alex, Q_REFUND)
        warm = cached(ask(alex, Q_REFUND))
        quiet = sync()
        claim(all(r["added"] == r["updated"] == r["removed"] == 0 for r in quiet.values()),
              "a sync with nothing to do changes nothing", str({k: v["unchanged"] for k, v in quiet.items()}))
        claim(warm and cached(ask(alex, Q_REFUND)), "…and does not empty the answer cache")

        status, chain = _call("GET", "/audit/verify", None, marcus)
        claim(status == 200 and chain.get("ok"), "the audit chain still verifies",
              f"{chain.get('rows_checked')} rows")
    finally:
        _author("reset")
        sync()

    claim(documents() == start, "the sources are back where they started", str(documents()))

    passed = sum(ok for ok, _, _ in rows)
    print(f"\n{passed}/{len(rows)} claims held")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
