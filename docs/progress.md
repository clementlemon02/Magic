# Internal Brain — progress

As of **26 September 2026**. Rationale lives in the Decision Log and in the PRs; this
is the state of the build, what was measured, and what is still open.

## 1. Where we are

The full request path runs end to end on the real stack (Ollama `qwen2.5:7b` +
`mxbai-embed-large`, Postgres 16 + pgvector), with no stand-ins:

| Component | State | Where |
|---|---|---|
| Router (rag / sql / clarify) | Done; distilled fast path in review (#21) | `src/agents/router.py` |
| Retrieval, ACL enforced in SQL (§1) + conflict detection | Done | `src/agents/retrieval.py` |
| SQL Tool (fixed templates, ACL predicate) | Done; the answer is rendered from the row, no model in the loop (#27) | `src/agents/sql_tool.py` |
| Synthesizer, Verifier | Done; confidence measured from the verdict token (#23) | `src/agents/synthesizer.py`, `verifier.py` |
| Escalation (generic refusal / audit-only reason, §5) | Done | `src/agents/escalation.py` |
| Hash-chained audit log + `verify_audit_chain()` (§6) | Done, tamper demo works | `src/agents/audit.py` |
| Compliance inquiry, revoke / grant, knowledge gaps | Done | `src/api/compliance.py` |
| Permission-aware answer cache | Done; cache hits are audited | `src/cache.py` |
| Constant-time refusal | Done; 0% of refusals escape the 4.0s deadline (#25) | `src/api/main.py` |
| Demo corpus (12 docs, 4 platforms) + seed script | Done | `src/connectors/`, `scripts/seed_demo.py` |
| UI | Done; four pages, one shared stylesheet and one client script (#28) | `src/api/ui.py`, `ask.html`, `dashboard.html`, `audit.html`, `sources.html` |
| Authentication | Done; signed bearer token, role re-read from the database (#28) | `src/api/auth.py` |
| CI | Done; the suite runs on every PR against a pgvector service (#28) | `.github/workflows/tests.yml` |

Tests: **175 pass** on `main` with `RUN_DATABASE_INTEGRATION=1`.

Lanes: Clement took over Jin Hui's workstream (escalation, audit, compliance,
knowledge-gap) on 25 Sep and Chris's open retrieval/corpus items on 26 Sep.

## 2. Running the demo

```bash
docker compose up -d                          # Postgres on POSTGRES_PORT (5433 locally)
ollama serve                                  # needs qwen2.5:7b and mxbai-embed-large
psql "$DATABASE_URL" -f src/db/schema.sql     # first time only
psql "$DATABASE_URL" -f scripts/seed_users.sql   # the eight personas, with passwords
.venv/bin/python -m scripts.seed_demo         # documents, grants, 600 transactions
.venv/bin/uvicorn src.api.main:app --port 8000
open http://localhost:8000
```

`/health` reports the database and the model. The server warms the model at startup
and `OLLAMA_KEEP_ALIVE` keeps it resident, because a cold model answers the first
question in ~12s and an idle one unloads after five minutes — which pushes the next
refusal past its 4.0s deadline, on the property the demo exists to show.

### Signing in

Every route takes a signed bearer token; there is no other way in. **Every persona's
password is `demo`.**

| | | |
|---|---|---|
| `alex.tan@aurelia.example` | support, clearance 0 | asks the questions in Scenarios 1–3 |
| `marcus.lim@aurelia.example` | compliance officer | the only role that can read a refusal's reason |

Sign in as one, then use **Add identity** in the header to hold the other as well. The
side-by-side comparison needs two tokens on purpose: showing what two people see means
being able to authenticate as both, not asking on anyone's behalf.

### The four pages

| Page | What it is for | Who |
|---|---|---|
| **Ask** `/` | One question across Confluence, Jira, Slack and Drive. You get an answer you are allowed to see, or one refusal. "Side by side" runs the same question as two callers. | anyone |
| **Gaps** `/dashboard` | Where the permission model is wrong. *Access gaps* are documents people wanted and could not see — fix the grant. *Knowledge gaps* are questions nothing answers — write the page. | officer |
| **Audit** `/audit` | Every request, hash-chained. The only place a refusal's real reason can be read back, and reading it is itself recorded. **Verify chain** walks the table. | officer |
| **Sources** `/sources` | What is connected, how fresh it is, and who may currently read what. Revoking a grant takes effect on the next question. | officer |

Three of the four are officer-only, so signing in as Alex and clicking around gets you
refused — that is the product working. The nav dims and locks those three when the
signed-in person cannot open them.

### What to try, in order

| | Do this | Expect |
|---|---|---|
| 1 — cross-platform answer | Ask *"How long do customers have to contest a chargeback?"* | 45 days, citing Confluence **and** Drive. **Show passage** opens the text the wording came from. |
| 2 — the headline | **Side by side**, ask *"What triggers an AML escalation review?"* | Alex refused on the 4.0s mark, Marcus answered with SGD 9,500 / Tier 2 / Project Nightingale. Both bars, one scale. |
| 3 — §1 on screen | Ask *"How many transactions were flagged for AML in August?"*, then **Show query** | The highlighted `acl_tags &&` line, with the caller's own tags. Alex 3, Marcus 5. |
| 4 — the two-tier refusal | As Marcus, **Audit** → a refused request | The explanation Alex never sees, the withheld items, and a `compliance_inquiry` row appended for having looked. |
| 5 — tamper evidence | Edit an `audit_log` row in psql, then **Verify chain** | Names the first broken row. Also `python -m src.agents.audit verify`, exit 1. |
| 6 — live revocation | As Marcus, **Sources** → revoke a grant, re-ask, then grant it back | Refused, then answered. The grant count moves with it. |
| 7 — where it is wrong | As Marcus, **Gaps** | Access gaps ranked by *distinct askers*; knowledge gaps clustered, e.g. "Parental Leave Policy". |

`scripts/seed_activity.py` asks a spread of questions as several personas, so the two
gap reports have something to show on a fresh database.

**Do not** run `TRUNCATE audit_log` casually: the chain is append-only by design.
The test suite no longer touches demo data (fixed in #16).

## 3. What was built this session (#13–#28)

| PR | What | Headline number |
|---|---|---|
| #13 | Constant-time refusal (the implementation that missed #12's merge) | refusal median ratio 0.49 → 1.000 on the old corpus |
| #14 | Escalation, audit hash chain, compliance APIs | tamper flagged at the exact row |
| #15 | Router stops clarifying clear-but-unanswerable questions | 11/16 → 16/16; demo questions added to evals |
| #16 | Reproducible seed; `all-staff` visibility; test no longer wipes demo data; `"None"` dept bug | SQL counts 0 → correct |
| #17 | SQL answers as sentences, citing the query | `5` → "5 transactions were flagged…" |
| #18 | Real corpus; conflict calibration eval; Verifier threshold rule | false conflicts 0/13 |
| #19 | Stop a hop that retrieved the same chunks as the last | 10/40 ungrounded refusals end a hop early |
| #20 | Time budget: no new hop after 2.5s | 3-hop refusals 10.4s → 4.01s |
| #21 *(open)* | Router distilled into an embedding classifier | 26/40 routed in ~23ms; 38/40 vs LLM 39/40 |
| #23 | Verifier confidence read from the verdict token, not self-reported | Brier 0.0743 → 0.0442; no extra latency |
| #24 | Query-time source recheck for restricted docs; access-gap report; 8 personas | staleness window closed; gaps ranked by distinct askers |
| #25 | Retrieval hop budget 2.5s → 2.0s, re-measured idle | refusals escaping 12.9% → **0.0%**, no answers lost |
| #26 | The web UI: asker's page at `/`, both gap reports at `/dashboard` | the API has a surface; refusal vs answer visible side by side |
| #27 | Verifier judges cited passages only; query answers skip it entirely | judge 6.80s → 2.54s; SQL path 5.7s → 1.8s |
| #28 *(open)* | The four-page UI, CI, per-hop audit rows, startup warm-up, adversarial routing, evidence and query views, sources admin, authentication | 273 tests, CI green; refusal escapes 12.9% → 0% |
| #31 *(open)* | `RETRIEVAL_MIN_SCORE` 0.55 → 0.65 (dropped off-topic chunks the Synthesizer read then ignored); extractive fast paths for the Synthesizer and Verifier — a near-literal single-passage answer skips both model calls entirely | one live question: 5.86s → 5.01s → **0.50s**, byte-identical answer |

## 4. Evals (all `.venv/bin/python -m evals.<name>`)

| Eval | What it measures | Latest |
|---|---|---|
| `run` | Router, Verifier, Synthesizer, SQL Tool, Clarification, prompt injection | Verifier 7/8 caught, 6/7 kept; **Router adversarial 4/4**; rest 100% |
| `leak_probe` | Prompt-based ACL baseline vs ours | baseline leaks; ours 0 |
| `conflict_calibration` | Conflict check on the seeded corpus | 8/9 raised, 0/13 false |
| `refusal_timing` (`--padded`) | Whether refusal timing reveals the cause | all four §12 criteria pass; 0% escape |
| `router_student` | Distilled Router vs teacher | see #21 |
| `confidence_calibration` | Verifier confidence: Brier, and a threshold sweep | measured 0.0982 vs self-reported 0.1360 (30 Sep; grows with `VERIFIER_CASES`) |
| `retrieval_threshold_sweep` | Recall vs noise across `RETRIEVAL_MIN_SCORE` candidates | recall holds 0.55→0.68, breaks at 0.70; set to 0.65 |
| `fast_path_sweep` | Synthesizer extractive shortcut: fires correctly vs fires wrong/on unanswerable | 0.25 sits above the highest measured miss (0.167), below the lowest hit (0.364) |
| `cache_probe` | Permission-aware cache | — |

## 5. Open issues

1. **Verifier false refusal — root-caused, fix built, deliberately not shipped live.**
   "Can I approve a SGD 3,000 refund myself?" is refused end to end even though the
   SGD 2,000 rule is stated nearly verbatim in `confluence:SUPPORT/refund-policy` and
   confirmed in two more passages. Per passage, three of four ground it cleanly
   (0.99-1.00); `drive:file-refund-playbook`'s phrasing ("above that ask your team
   lead", an instruction rather than a stated fact) refuses alone, and poisons every
   combination it appears in — the joint refusal gets MORE confident as more passages
   are added, not less. `_recheck_each_passage` (`allow_recheck=True`) fixes it: ask
   per passage, take the first that grounds. **Not wired into `verify_node`.** Measured
   worst case ~13s per verification, and a full `refusal_timing.py --padded` sweep with
   it unconditional took refusals escaping the 4.0s deadline from a documented 0% to
   5.7% against the ≤1% target (§12) — a regression on the one non-negotiable property
   to fix a single, rarer, fails-CLOSED accuracy edge case. `evals/run.py` opts in
   (not deadline-bound); the live graph does not. Verifier — grounded answers kept:
   6/7 → 7/7 in the eval, unchanged in production.
2. **Known limits, documented in code:** the audit chain cannot detect truncation of
   its newest rows; a revoked user's refusals on internal documents can appear as
   knowledge gaps; `AUTH_SECRET` is a committed demo key with no refresh, revocation
   list or rotation — whoever holds it can mint a token for any user.

Closed in #29: **the Verifier fail-open**, the only open issue that failed in the dangerous
direction. A `grounded` verdict is now checked against the figures in code — every number the
answer states must appear in the evidence or the question, or the verdict is downgraded
(`verifier.unsupported_quantities`, one direction only). Hallucinations caught 7/8 → 8/8. The
check is precise about it: on the same answer with all four passages present, where SGD 2,000
genuinely appears, it does not fire — so it never masked the false refusal above; that one
needed its own, separate fix (§5.1), which exists but ships gated off for the reason stated there.

Also closed in #29: the `clarify` dumping ground (every non-question — greetings, thanks,
"write me a poem" — was answered with a clarifying question), and cache hits being audited with
`initial_state`'s `rag` placeholder instead of the route that produced the entry.

Caught before shipping in #29: `decline` was first added as a fourth option in the routing
prompt, which took `ROUTER_ADVERSARIAL` from 4/4 to 2/4 — a plain company question wrapped in an
instruction was declined outright. Moved to a separate yes/no asked only on the `clarify` branch;
back to 4/4. Worth remembering as a shape: a 7B has a budget for how many distinctions one prompt
can carry, and the cost of spending it lands on whatever that prompt was already doing.

Closed in #28: the router adversarial miss (2/4 through the path the graph actually
runs, now 4/4 — the eval had been scoring the teacher rather than `route_node`);
multi-hop requests recording only their last hop; and `X-User-Id`, a header anyone
could set, which gated seven officer surfaces.

## 6. Next

1. Verifier rework, **partly done, rest to be rescoped**. Two halves shipped in #27:
   the judge now reads only the cited passages (6.80s -> 2.54s on the four-passage
   case), and a query-only answer is rendered from the result row with no model in the
   loop at all (5.7s -> 1.8s), which removes the judge from that path rather than
   speeding it up. The remaining accuracy work was planned against the belief that too
   many passages were the cause, which §5 issue 2 now shows is wrong — and issue 1 is a
   different problem again: a judge that accepts unrelated evidence is not a judge
   confused by too much of it. Atomic claim decomposition plausibly targets both;
   measure before committing to it.
2. Set `VERIFIER_CONFIDENCE_THRESHOLD` off the calibration curve. #23 made the number
   real — measured from the verdict token, Brier 0.0442 against the self-reported
   0.0743 — and `evals/confidence_calibration.py` sweeps it: at 0.6 the threshold
   catches nothing, and [0.8, 0.9] catches the issue-2 false refusal while needlessly
   escalating 0 of the 13 correct verdicts. **Not raised yet**: that band rests on one
   wrong verdict out of 14. Label more cases, re-run, then set it.
3. Optional: Jev (TypeSafe) as a second Router backend if access arrives. Good fit for
   routing, poor fit for the Verifier (its documented weaknesses — arithmetic, long
   irrelevant state, injected instructions — are exactly our Verifier's failure cases,
   and it would send document content to a US cloud). Waitlisted. Use `typesafe.ai` /
   `docs.typesafe.ai` only; `jevapi.org` and `jevtypesafe.org` are lookalikes.

   **`system-one-adapter-python` was evaluated on 26 Sep and rejected for the critical
   path.** It is TypeSafe's own MIT-licensed shim exposing the Choice/Score/Noul
   interface over OpenAI/Anthropic/Gemini, and its README states its purpose: comparing
   TypeSafe against an LLM on cost, speed and intelligence. It is a benchmark harness,
   not a decision layer. Three reasons it does not fit:

   - Its `probabilities` mode is **self-reported**: `_schema.py` puts one `[0, 1]` float
     per label in the output JSON schema and the model writes the numbers itself, which
     is why `normalize_probabilities` exists at all. That is the failure #23 had to fix,
     and it fixed it by reading token logprobs — which this adapter does not expose.
   - Every call is an LLM round-trip plus schema validation plus corrective retries. The
     distilled Router (#21) decides in ~23ms with no LLM, and refusal latency was
     already the tightest budget in the system.
   - It hard-depends on `typesafe-sdk`, so it is not the interface without the cloud
     dependency. Local Ollama would work only through `OpenAIProvider(base_url=...)`,
     which is an escape hatch rather than a supported path, and is untested here.

   It is a well-built library (typed, tach-enforced boundaries, network-blocked tests)
   and the Choice/Score/Noul framing — a decision, not a generation — is the good idea
   we already took into #21. Revisit it **only** if Jev access arrives and we want an
   honest A/B of Jev against qwen2.5 on one interface. That is an `evals/` experiment,
   never a `src/` dependency.

## 7. Owner actions outside the code

- **CodeBuddy / WorkBuddy proof** — 3+ screenshots or recordings. Mandatory; without
  it the project is not scored.
- **Project Description** — due 12 Oct.
- **UI** — not started (Miora).
- **Hackathon rules** — confirm an external model API is allowed before any Jev use.
