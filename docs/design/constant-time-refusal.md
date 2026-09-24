# Constant-time refusal

**Status:** proposed — baseline measured, not implemented
**Owner:** Clement (orchestration)
**Touches:** `src/api/main.py` (§8 shared), `src/config.py`, `.env.example` (§8 shared)
**Baseline:** `evals/results/refusal_timing_baseline.json`, commit `b641922`

---

## 1. Problem statement

Internal Brain refuses in one fixed sentence whatever the cause, so that a caller can't tell *"there is an answer you may not see"* from *"there is no answer."* That property is the product's central security claim, and it is currently **true for the bytes and false for the clock.**

A refusal caused by a permission conflict returns faster than one caused by insufficient evidence, because the conflict short-circuits the graph straight to Escalation, while an ungrounded question runs up to three retrieval-and-synthesis hops first. The two can be told apart from a single timing measurement.

That turns the refusal into a **timing oracle for the existence of restricted documents.** A support user who can't read the AML procedure can still find out that one exists — and what it is about — by asking questions and timing the refusals. This is the same attack class as [BudgetLeak](https://arxiv.org/pdf/2511.12043), which infers RAG membership from a generation-budget side channel rather than the response text.

The threat actor is the ordinary authenticated asker, which is exactly who the refusal exists to withhold from.

## 2. Objective

Make a refusal indistinguishable **by time** as well as by content, so that no timing measurement a caller can take separates a permission-conflict refusal from any other refusal — without changing the latency of answers or clarifications.

Stated as a measurable target, on the baseline hardware:

- the conflict and non-conflict refusal latency distributions **overlap** (they currently don't);
- the ratio of their medians is **within 5% of 1.0** (currently 0.49);
- the fraction of refusals whose natural latency exceeds the deadline — the residual leak — is **≤ 1%**, and reported rather than assumed to be zero;
- answer and clarification latency is **unchanged** within run-to-run noise;
- the refusal text remains **byte-identical**.

## 3. Impact

**Security.** Closes the one channel known to reveal restricted-document existence to an authenticated caller. Without it, the claim that a refusal "reveals nothing" does not survive a stopwatch.

**Credibility of the headline claim.** The pitch, the leak probe, the architecture diagram and several PRs currently describe the refusal as "byte-identical to having no answer." That is true, and it implies more than it establishes. This change is what makes the stronger reading defensible — and measuring our own claim, finding it false, and fixing it is a stronger story than the claim was.

**Rubric.** Feeds *Responsible AI & Ethics* directly (a privacy defence with before-and-after numbers), *Innovation & Creativity* (see §5 for exactly what is and isn't new), and *Technical Execution*.

**Cost to the user.** Every refusal gets slower. That is the entire trade, and it is stated rather than hidden: *refusals are deliberately slower so they stop telling you why.*

## 4. Evidence — baseline measurement

`evals/refusal_timing.py`: 110 requests through the real `/query` endpoint — caller resolution against Postgres, real ACL-filtered pgvector retrieval, real model — interleaved in a seeded random order so drift can't favour a class. Each request is classified by **what actually happened**, read from the final graph state, not by the question's intended category. Answer cache disabled so repeated answers aren't served at 0.03s. Apple M3 Pro, 18 GB, `qwen2.5:7b` chat, `mxbai-embed-large` embeddings, 8-document corpus.

| observed path | n | min | p50 | p95 | max |
|---|---|---|---|---|---|
| refused — permission conflict | 30 | 0.18s | **1.23s** | 1.25s | **1.36s** |
| clarified | 20 | 1.00s | 1.75s | 2.08s | 2.09s |
| refused — ungrounded, 3 hops | 40 | **1.44s** | **2.50s** | 2.79s | 2.99s |
| answered | 20 | 2.92s | 3.75s | 3.93s | 4.10s |

- **Distinct refusal texts: 1.** The bytes are identical — the existing claim holds as far as it goes.
- **Distributions do not overlap.** The slowest conflict refusal (1.36s) is faster than the fastest ungrounded one (1.44s). A threshold at ~1.4s classifies all 70 refusals correctly.
- **The median gap is 2.0×**, not the 16.7× an earlier five-run probe suggested. That probe ran the same question consecutively, which kept the inference engine's prompt cache warm — the effect described next.
- A first run (same commit, earlier classifier) produced the same separation, with the slowest refusal at **3.19s**. That figure sets the deadline headroom in §7.

### A second timing channel, found during measurement

The conflict path's latency is bimodal — minimum 0.18s, median 1.23s, almost nothing between. Both models were confirmed resident on the GPU, so it isn't model swapping. Splitting by the *preceding* request:

| conflict refusal, when the previous request was… | n | median | range |
|---|---|---|---|
| another conflict refusal | 7 | **0.41s** | 0.18–0.43s |
| anything else | 23 | **1.23s** | 1.23–1.36s |

Replicated across both runs — identical medians, near-identical ranges. The Router prompt has a long constant prefix that Ollama caches; any request that runs synthesis in between evicts it. **So a request's latency depends on what the previous request was — including another user's.** That is the multi-tenant KV-cache channel [KVGov](https://arxiv.org/abs/2608.09225) addresses, operating at the inference-engine layer and independent of the graph. Padding absorbs it for refusals; it remains for answers (§9).

### Paths not observed

Two of the four routes into Escalation (`graph.py:59, 67, 72, 77`) appeared. On an 8-document corpus with `RETRIEVAL_MIN_SCORE=0.55`, every question retrieves *something*, so the **no-evidence** path never ran, and the **low-confidence** path never triggered. Both must be covered once the corpus grows (§11).

## 5. Prior art, and what is actually new

**Known:** timing side channels in LLM serving ([Early Bird](https://arxiv.org/html/2409.20002v1), [SpliceLeak](https://arxiv.org/html/2606.21842)); semantic-cache timing attacks, with 43–100% input extraction on legal services ([InputSnatch](https://arxiv.org/abs/2411.18191)); membership inference over RAG corpora ([Is My Data in Your Retrieval Database?](https://arxiv.org/abs/2405.20446), [BudgetLeak](https://arxiv.org/pdf/2511.12043)); cache partitioning against cross-tenant leakage ([KVGov](https://arxiv.org/abs/2608.09225)); and **constant-time execution as a mitigation**, which InputSnatch names explicitly.

**Not claimed as new:** the technique. Padding to a constant time is standard in cryptography and already proposed for LLM serving.

**What appears to be specific to this work:** the leak itself — refusal-path timing in access-controlled RAG, where the secret is whether restricted documents *exist*, and the cause is application control flow rather than the inference engine. No published measurement of it turned up.

**Search limits, stated:** three Google Scholar queries and one Semantic Scholar pass, rate-limited on four of five queries; abstracts and snippets only. A thorough prior-art check needs an API key and full-text reading of the closest four or five papers.

**Framing for the pitch:** *we measured a leak the literature describes in general, in a setting it hasn't measured, and applied a known mitigation with before-and-after numbers.*

## 6. Design

### 6.1 Pad to a fixed deadline

Every refusal returns no earlier than `REFUSAL_DEADLINE_SECONDS` after the request arrived. If the graph finishes sooner, the response waits out the difference.

### 6.2 Pad only refusals

| response | padded | why |
|---|---|---|
| refusal (`escalated=True`) | **yes** | the channel |
| answer | no | answer vs refusal is visible in the text already; padding costs latency on every answer for nothing |
| clarification | no | decided by the Router **before** retrieval, so its timing cannot depend on restricted content |
| error (5xx) | no | a different status code is already distinguishable, and an outage should report fast |

### 6.3 Measure from request arrival, at the API boundary

The clock starts when `/query` begins handling the request, not when the graph starts. Caller resolution and the answer-cache lookup both run before the graph; starting later would leave their variation outside the envelope. Padding lives at the boundary the response leaves through, not inside the graph.

### 6.4 The deadline is configured, never learned

A deadline derived from recently observed latency would drift with what is being asked, and the drift would itself leak. One fixed value, documented as **per-model and per-hardware** — the third such constant, alongside `RETRIEVAL_MIN_SCORE` and `QUERY_CACHE_SIMILARITY`.

### 6.5 Sleep asynchronously

`/query` is currently a sync handler, which Starlette runs on a threadpool of about 40 workers. A `time.sleep` would hold one for seconds per refusal; firing restricted questions in parallel would exhaust the pool, turning the defence into a denial-of-service amplifier.

The handler becomes `async`, runs the graph via `run_in_threadpool`, and pads with `asyncio.sleep`, which yields the event loop rather than a thread.

### 6.6 Injectable clock

The clock and sleeper are parameters of `create_app`, so tests assert exact behaviour without sleeping.

## 7. Choosing the deadline

Fraction of refusals whose natural latency would still exceed each candidate deadline (baseline run, 70 refusals):

| deadline | all refusals exceeding | non-conflict exceeding | conflict refusal cost |
|---|---|---|---|
| 2.0s | 41.4% | 72.5% | +0.8s |
| 3.0s | 0.0% | 0.0% | +1.8s |
| **4.0s** | **0.0%** | **0.0%** | **+2.8s** |
| 5.0s | 0.0% | 0.0% | +3.8s |

A refusal arriving *after* the deadline is necessarily not the fast conflict path, so exceedance is a **one-directional leak**: it can confirm a question had no restricted answer. That is why the rate is reported, not assumed away.

**Recommendation: 4.0s.** 3.0s clears this run with only 0.01s to spare over the slowest refusal (2.99s), and the first run's slowest was 3.19s — it would have leaked. 4.0s leaves ~0.8s of headroom over the worst of 140 observed refusals across both runs. Answers are unaffected either way.

**The decision is yours:** a lower deadline keeps refusals snappier in the demo at the price of a measurable residual leak. The curve is the argument for whichever you choose.

## 8. Trade-offs

| gain | cost |
|---|---|
| refusal no longer reveals whether restricted content exists | conflict refusals slow from 1.23s to 4.0s |
| absorbs the prefix-cache channel for refusals too | refusals become slower than answers (3.75s median) — visible, but harmless, since answer vs refusal is visible anyway |
| a defensible version of the headline claim | one more per-hardware constant to re-measure |
| no extra model cost | each padded refusal holds an open connection for up to 4s |

## 9. Alternatives considered

| alternative | verdict |
|---|---|
| **Dummy work** — after a conflict, run synthesis and verification anyway, then discard | Higher fidelity: matches the *variance* of the slow path, not only its median. Roughly triples model calls on every restricted question and is harder to reason about. **Follow-up**, not first. |
| **Random jitter** | Rejected on its own. An attacker averages repeated queries and the signal returns. |
| **Pad every response** | Rejected. Hides a distinction the asker can already see, at a latency cost on every answer. |
| **Per-cause deadlines** | Rejected. Which deadline applied is itself the leak. |
| **Round up to time buckets** | Weaker than a fixed deadline — leaks the bucket — but degrades more gracefully in the tail. Worth combining if tail exceedance proves hard to control. |

## 10. Out of scope, named deliberately

- **Prefix-cache channel on answers.** §4. Answer latency still depends on the previous request, including another user's. The reference fix is per-principal cache salting at the inference layer ([KVGov](https://arxiv.org/abs/2608.09225)), which belongs in the serving stack rather than the application.
- **Answer-cache peer-activity leak.** Within a permission partition, a hit at 0.03s against a 3.75s miss reveals that a peer with identical access recently asked the same thing — InputSnatch's attack. Refusals are never cached, so this change doesn't touch it. Candidate fix: k-anonymous partitions, sharing only once *k* users hold an identical fingerprint.
- **Network-level channels.** Packet sizes and inter-token timing. The refusal is fixed-length and responses aren't streamed, so these should already be closed. Not measured.

## 11. Implementation

1. **Config** — `REFUSAL_PADDING_ENABLED` (default on) and `REFUSAL_DEADLINE_SECONDS=4.0` in `src/config.py` and `.env.example`, documented as per-hardware.
2. **`/query` becomes async** — graph via `run_in_threadpool`; clock started before caller resolution and the cache lookup.
3. **Padding** — if `escalated`, `await asyncio.sleep(max(0, deadline − elapsed))`. Clock and sleeper injectable through `create_app`.
4. **Tests** — see §12.
5. **Eval** — `evals/refusal_timing.py` gains a padding on/off switch; after-run written beside the baseline.
6. **Correct the claim** — leak probe docstring, architecture diagram card and the relevant PR text, from "byte-identical" to what has been measured.
7. **Before code freeze, once the corpus grows** — re-run the measurement so the no-evidence and low-confidence paths are exercised, and re-derive the deadline.

## 12. Verification

**Unit tests, injected clock:**

- a refusal completes at exactly the deadline;
- an answer and a clarification are not delayed;
- a refusal already past the deadline is not delayed further, and never sleeps a negative interval;
- a conflict refusal and a hop-cap refusal report identical durations;
- the refusal text stays byte-identical;
- **concurrency:** 50 simultaneous refusals don't delay a concurrent answer by anything close to the deadline — the async sleep holds no worker.

**Eval, real stack — success means all of these:**

| measure | baseline | target |
|---|---|---|
| conflict vs non-conflict distributions overlap | no | **yes** |
| median ratio, conflict ÷ non-conflict | 0.49 | **0.95–1.05** |
| refusals exceeding the deadline | — | **≤ 1%**, reported |
| answer p50 | 3.75s | unchanged ± noise |
| clarification p50 | 1.75s | unchanged ± noise |
| distinct refusal texts | 1 | **1** |

## 13. Risks

- **The deadline is hardware-bound.** A slower demo machine, a larger model or a bigger corpus moves the tail. The deadline must be re-measured, not copied. The eval makes that cheap.
- **Small sample.** 70 refusals per run. "0% exceeding 4.0s" means none of 140 across two runs, not a guarantee. More runs tighten it.
- **Connection pressure.** Async sleep prevents thread starvation but not open-connection count. Per-caller rate limiting is the backstop if it matters.
- **Only two of four refusal causes measured** (§4). The unmeasured ones could be faster than the conflict path; padding covers them regardless, but the baseline doesn't prove it.

## 14. Decision needed

- **The deadline value.** Recommendation: **4.0s**, per §7.
- **Whether to pursue dummy work as a follow-up**, for variance-matched rather than median-matched timing.
