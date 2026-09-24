"""Can a stopwatch tell *why* the system refused?

    LLM_BACKEND=ollama .venv/bin/python -m evals.refusal_timing

Times requests end to end through the real /query endpoint — caller resolution,
real retrieval against pgvector, real model — and classifies each one by what
actually happened, read from the final state the audit node receives. Classes are
observed, never assumed from the question.

The answer cache is disabled for the run: answers repeat, and a cache hit at 0.03s
would contaminate the answer timings this is meant to measure.

Output is the baseline for constant-time refusal (docs/design/constant-time-refusal.md):
per-class latency, whether the permission-conflict refusal overlaps the others, and
the fraction of refusals that would still exceed each candidate deadline.
"""

import json
import os
import platform
import random
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

os.environ["QUERY_CACHE_ENABLED"] = "false"  # before any settings are read

from fastapi.testclient import TestClient  # noqa: E402

from src.agents.clarification import clarify_node  # noqa: E402
from src.agents.retrieval import retrieval_node  # noqa: E402
from src.agents.router import route_node  # noqa: E402
from src.agents.sql_tool import sql_tool_node  # noqa: E402
from src.agents.synthesizer import synthesize_node  # noqa: E402
from src.agents.verifier import verify_node  # noqa: E402
from src.api.main import create_app, load_user  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.llm.factory import get_chat_model  # noqa: E402

RUNS_PER_QUESTION = 10
CANDIDATE_DEADLINES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

QUESTIONS = {
    # Expected to hit a restricted match. Classified by outcome, not by this label.
    "What triggers an AML escalation review?": "restricted",
    "What is the AML control gap?": "restricted",
    "What does the AML evidence register contain?": "restricted",
    # Retrieves permitted refund material that does not actually answer them.
    "How long do customers have to contest a chargeback?": "ungrounded",
    "Who approves refunds above the limit?": "ungrounded",
    "What fee is charged on a refund?": "ungrounded",
    # Nothing in the corpus is about these at all.
    "What is our parental leave policy?": "no-evidence",
    "Where is the office cafeteria?": "no-evidence",
    "What is the company's stock ticker?": "no-evidence",
    # Reference: permitted questions the corpus answers.
    "How long does support have to review a refund request?": "answer",
    "What happened with the refund workflow incident?": "answer",
}


def classify(state: dict) -> str:
    """Name the path the request actually took."""
    if not state.get("escalated"):
        # Clarification ends the turn with a question, not an answer. The first run
        # lumped the two together and reported 40 "answers" from 20 answerable runs.
        if state.get("clarification_question") and not state.get("draft_answer"):
            return "clarified"
        return "answered"
    if state.get("permission_conflicts"):
        return "refused: permission conflict"
    verification = state.get("verification")
    hops = state.get("hop_count", 0)
    if verification is not None and verification.grounded:
        return "refused: low confidence"
    if not state.get("retrieved_chunks"):
        return f"refused: no evidence ({hops} hops)"
    return f"refused: ungrounded ({hops} hops)"


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> int:
    settings = get_settings()
    chat = get_chat_model()
    final_states: list[dict] = []

    nodes = {
        "router": partial(route_node, chat_model=chat),
        "clarification": partial(clarify_node, chat_model=chat),
        "retrieval": retrieval_node,
        "sql_tool": partial(sql_tool_node, chat_model=chat),
        "synthesizer": partial(synthesize_node, chat_model=chat),
        "verifier": partial(verify_node, chat_model=chat),
        # Jin Hui's two nodes are still stubs. These stand-ins do no work, so they
        # add nothing to the timings; the audit one captures the final state.
        "escalation": lambda s: {"escalated": True},
        "audit": lambda s: (final_states.append(dict(s)), {"audit_events": []})[1],
    }
    client = TestClient(create_app(nodes=nodes, user_loader=load_user))

    for _ in range(2):  # load model weights before anything is timed
        client.post("/query", json={"query": "warm up", "user_id": 1})

    schedule = [q for q in QUESTIONS for _ in range(RUNS_PER_QUESTION)]
    random.Random(7).shuffle(schedule)  # interleave, so drift can't favour one class

    samples: list[dict] = []
    for i, question in enumerate(schedule, 1):
        before = len(final_states)
        started = time.perf_counter()
        response = client.post("/query", json={"query": question, "user_id": 1})
        elapsed = time.perf_counter() - started
        state = final_states[before] if len(final_states) > before else {}
        samples.append({
            "question": question,
            "class": classify(state),
            "seconds": elapsed,
            "status": response.status_code,
            "text": response.json().get("text", "")[:60] if response.status_code == 200 else "",
        })
        print(f"\r  {i}/{len(schedule)}", end="", flush=True, file=sys.stderr)
    print(file=sys.stderr)

    by_class: dict[str, list[float]] = {}
    for s in samples:
        by_class.setdefault(s["class"], []).append(s["seconds"])

    refusal_texts = {s["text"] for s in samples if s["class"].startswith("refused")}
    conflict = by_class.get("refused: permission conflict", [])
    others = [
        t for c, ts in by_class.items()
        if c.startswith("refused") and c != "refused: permission conflict"
        for t in ts
    ]
    all_refusals = conflict + others

    print(f"\nbackend {settings.llm_backend} · {settings.ollama_model} · "
          f"{settings.ollama_embedding_model} · {platform.machine()} "
          f"· {len(samples)} requests\n")
    print(f"{'observed path':<34}{'n':>4}{'min':>8}{'p50':>8}{'p95':>8}{'max':>8}")
    print("-" * 70)
    for cls in sorted(by_class, key=lambda c: pct(by_class[c], 0.5)):
        ts = by_class[cls]
        print(f"{cls:<34}{len(ts):>4}{min(ts):>7.2f}s{pct(ts,.5):>7.2f}s"
              f"{pct(ts,.95):>7.2f}s{max(ts):>7.2f}s")

    print(f"\ndistinct refusal texts: {len(refusal_texts)}  (1 means byte-identical)")
    if conflict and others:
        gap = min(others) - max(conflict)
        print(f"conflict refusals max {max(conflict):.2f}s · other refusals min {min(others):.2f}s")
        print("distributions overlap: "
              + ("no — a single measurement separates them" if gap > 0 else "yes"))

    print(f"\n{'deadline':>9}  {'refusals exceeding it':>22}  {'non-conflict exceeding':>24}")
    curve = []
    for d in CANDIDATE_DEADLINES:
        a = sum(t > d for t in all_refusals) / len(all_refusals) if all_refusals else 0
        o = sum(t > d for t in others) / len(others) if others else 0
        curve.append({"deadline": d, "all_refusals_exceeding": a, "non_conflict_exceeding": o})
        print(f"{d:>8.1f}s  {a:>21.1%}  {o:>23.1%}")

    out = Path(__file__).parent / "results" / "refusal_timing_baseline.json"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out.write_text(json.dumps({
        "commit": commit,
        "machine": platform.machine(),
        "chat_model": settings.ollama_model,
        "embedding_model": settings.ollama_embedding_model,
        "runs_per_question": RUNS_PER_QUESTION,
        "classes": {c: {"n": len(ts), "min": min(ts), "p50": pct(ts, .5),
                        "p95": pct(ts, .95), "max": max(ts)} for c, ts in by_class.items()},
        "distinct_refusal_texts": len(refusal_texts),
        "deadline_curve": curve,
        "samples": samples,
    }, indent=2))
    print(f"\nwrote {out.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
