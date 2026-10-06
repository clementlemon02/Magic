"""Sweep SYNTHESIS_FAST_PATH_MIN_OVERLAP: how often the extractive shortcut fires,
and whether it ever fires on the wrong answer.

Companion to evals/retrieval_threshold_sweep.py. The Synthesizer's fast path
(src/agents/synthesizer.py:_extractive_answer) returns one of a retrieved chunk's
own sentences UNEDITED when it clearly leads every other candidate on query-word
overlap, skipping the model call entirely. The Verifier's fast path
(src/agents/verifier.py:_verbatim_chunk) then finds that exact text sitting inside
the same chunk and skips its own model call too — so a hit here is worth close to
the full Synthesizer-plus-Verifier latency, not just one of them.

    LLM_BACKEND=ollama .venv/bin/python -m evals.fast_path_sweep

Two things must hold at whatever ratio is chosen. It must never fire on an
unanswerable question: SYNTHESIZER_CASES hands every question the full fixture
CORPUS regardless of whether anything in it is on topic, specifically to test
that INSUFFICIENT still comes back when nothing answers the question — a fast
path that fires anyway on coincidental word overlap would hand back a confident,
wrong answer instead. And where it does fire on an answerable question, the
sentence must come from a chunk in that question's expected_docs, or it is
confidently citing the wrong source. The compound PII/on-call question never
fires at any candidate here — not because of the ratio, but because
`_looks_compound` excludes it structurally (see its docstring): a sentence
answering half of it scored close enough to a genuine single-answer hit,
depending only on phrasing, that no ratio in this sweep separated them safely.

Per CLAUDE.md's own convention for a tunable like this ("per-model; see
.env.example, re-measure if X changes") — this is Clement's call, not mine. This
sweep exists so that call is made from numbers.
"""

import sys

from evals.cases import SYNTHESIZER_CASES
from src.agents.synthesizer import _extractive_answer

CANDIDATES = [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50]


def main() -> int:
    header = f"{'ratio':>6}  {'fires':>6}  {'correct doc':>12}  {'wrong doc':>10}  {'fired on unanswerable':>22}"
    print(header)
    print("-" * len(header))

    bad_at = None
    for ratio in CANDIDATES:
        fires = correct = wrong = bad_fire = 0
        for question, chunks, expected_docs in SYNTHESIZER_CASES:
            result = _extractive_answer(question, chunks, ratio)
            if result is None:
                continue
            fires += 1
            _, source = result
            if expected_docs is None:
                bad_fire += 1
            elif source.document_id in expected_docs:
                correct += 1
            else:
                wrong += 1
        print(f"{ratio:>6.2f}  {fires:>6}  {correct:>12}  {wrong:>10}  {bad_fire:>22}")
        if (wrong or bad_fire) and bad_at is None:
            bad_at = ratio

    print("\nlowest ratio that fires, per case (blank = never fires in this sweep):")
    for question, chunks, expected_docs in SYNTHESIZER_CASES:
        print(f"\n  {question}  (expected docs: {expected_docs})")
        fired = False
        for ratio in sorted(CANDIDATES):
            result = _extractive_answer(question, chunks, ratio)
            if result is not None:
                sentence, source = result
                print(f"    fires from {ratio:.2f}: doc {source.document_id}: {sentence!r}")
                fired = True
                break
        if not fired:
            print(f"    never fires (falls through to the model at every candidate ratio)")

    if bad_at is not None:
        print(f"\nfirst ratio with a wrong or unanswerable fire: {bad_at}")
    else:
        print("\nno candidate ratio ever fired wrong or on an unanswerable question")
    print("\nThis is Clement's call — CLAUDE.md's own convention for this constant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
