"""The distilled Router against its teacher, on questions neither was trained on.

    .venv/bin/python -m evals.router_student

Three configurations over every labelled Router case in evals/cases.py:
  student  — the classifier alone, always taking its top route
  gated    — the classifier when p >= ROUTER_STUDENT_MIN_CONFIDENCE, else the LLM
  llm      — the LLM Router alone (the teacher)

Also reports calibration: a Brier score, and accuracy within confidence bands, so
"0.9 confident" can be checked against how often 0.9 is actually right.
"""

import time
import warnings
from functools import partial

warnings.filterwarnings("ignore")

from evals.cases import ROUTER_ADVERSARIAL, ROUTER_CASES, ROUTER_UNANSWERABLE  # noqa: E402
from src.agents.router import SELECTABLE_ROUTES, classify, student_probabilities  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.llm.factory import get_chat_model, get_embeddings  # noqa: E402

SETS = {
    "clear": ROUTER_CASES,
    "clear-vs-vague": ROUTER_UNANSWERABLE,
    "adversarial": [(q, want) for q, want, _ in ROUTER_ADVERSARIAL],
}


def main() -> int:
    threshold = get_settings().router_student_min_confidence
    chat, emb = get_chat_model(), get_embeddings()
    emb.embed_query("warm up")
    classify("warm up", chat_model=chat)

    rows, brier, bands = [], [], {}
    for name, cases in SETS.items():
        for q, want in cases:
            t = time.perf_counter()
            p = student_probabilities(q, emb)
            t_student = time.perf_counter() - t
            student_route, conf = max(p.items(), key=lambda kv: kv[1])

            t = time.perf_counter()
            llm_route = classify(q, chat_model=chat)
            t_llm = time.perf_counter() - t

            gated = student_route if conf >= threshold else llm_route
            t_gated = t_student + (0 if conf >= threshold else t_llm)
            rows.append((name, q, want, student_route, conf, gated, llm_route,
                         t_student, t_gated, t_llm))
            brier.append(sum((p[r] - (r == want)) ** 2 for r in SELECTABLE_ROUTES))
            band = "≥0.9" if conf >= 0.9 else "0.7–0.9" if conf >= 0.7 else "<0.7"
            bands.setdefault(band, []).append(student_route == want)

    def acc(sel, col):
        hits = [r for r in rows if sel(r)]
        return f"{sum(r[col] == r[2] for r in hits)}/{len(hits)}"

    print(f"threshold {threshold}\n")
    print(f"{'set':<16}{'student':>10}{'gated':>10}{'llm':>10}{'student decides':>18}")
    for name in SETS:
        sel = partial(lambda n, r: r[0] == n, name)
        mine = [r for r in rows if sel(r)]
        decided = sum(r[4] >= threshold for r in mine)
        print(f"{name:<16}{acc(sel, 3):>10}{acc(sel, 5):>10}{acc(sel, 6):>10}"
              f"{f'{decided}/{len(mine)}':>18}")

    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    print(f"\nlatency p50   student {med([r[7] for r in rows]) * 1000:.0f}ms · "
          f"gated {med([r[8] for r in rows]) * 1000:.0f}ms · llm {med([r[9] for r in rows]) * 1000:.0f}ms")
    print(f"Brier score   {sum(brier) / len(brier):.3f}  (0 is perfect, 0.667 is a uniform guess)")
    for band in ("≥0.9", "0.7–0.9", "<0.7"):
        if band in bands:
            ok = bands[band]
            print(f"  confidence {band:<8} right {sum(ok)}/{len(ok)}")

    print("\nstudent errors (conf, want, got):")
    for name, q, want, s_route, conf, gated, *_ in rows:
        if s_route != want:
            marker = "gated-out" if conf < threshold else "SHIPPED"
            print(f"  {conf:.2f} {want:<8}{s_route:<8} {marker:<10} {q[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
