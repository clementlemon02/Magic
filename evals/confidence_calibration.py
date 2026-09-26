"""Is the Verifier's confidence worth thresholding on? Measured on the labelled cases.

    .venv/bin/python -m evals.confidence_calibration

§4 sends a low-confidence verdict to Escalation. That only works if `confidence`
tracks whether the verdict is actually right. Two numbers say whether it does:

  Brier   mean (confidence - correct)^2 over the labelled cases, lower is better.
          A judge that says 1.0 every time scores exactly its own error rate and
          has told you nothing — that was the self-reported number's problem.
  Sweep   at each candidate threshold, how many WRONG verdicts it catches (those
          escalate instead of shipping) against how many RIGHT verdicts it
          needlessly escalates. VERIFIER_CONFIDENCE_THRESHOLD is picked off this,
          not guessed.

Both numbers are reported for the self-reported confidence and the measured one,
so the comparison is like for like on identical judgements.
"""

import json
import sys
from pathlib import Path

from evals.cases import VERIFIER_CASES
from src.agents.verifier import (
    PROMPT_TEMPLATE,
    _grounded_probability,
    _parse,
    _render_evidence,
)
from src.config import get_settings
from src.llm.factory import chat_with_logprobs

RESULTS = Path(__file__).parent / "results" / "confidence_calibration.json"
SWEEP = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]


def _brier(rows: list[dict], key: str) -> float:
    return sum((r[key] - r["correct"]) ** 2 for r in rows) / len(rows)


def main() -> int:
    rows = []
    for question, answer, evidence, should_be_grounded in VERIFIER_CASES:
        prompt = PROMPT_TEMPLATE.format(
            evidence=_render_evidence(evidence, None), query=question, answer=answer
        )
        pair = chat_with_logprobs(prompt)
        if pair is None:
            print("backend cannot report logprobs; set LLM_BACKEND=ollama")
            return 2
        raw, tokens = pair
        parsed = _parse(raw)
        measured = _grounded_probability(tokens)
        rows.append({
            "question": question,
            "answer": answer,
            "n_passages": len(evidence),
            "expected": should_be_grounded,
            "verdict": parsed.grounded,
            "correct": int(parsed.grounded == should_be_grounded),
            "self": parsed.confidence,
            # None means the verdict token was unreadable; treat as fully certain so
            # the measured column is never flattered by dropping hard cases.
            "measured": 1.0 if measured is None else measured,
        })

    wrong = [r for r in rows if not r["correct"]]
    print(f"{len(rows) - len(wrong)}/{len(rows)} verdicts correct\n")

    print(f"{'':<10}{'Brier':<10}(lower is better)")
    for key in ("self", "measured"):
        print(f"{key:<10}{_brier(rows, key):<10.4f}")

    print(f"\n{'threshold':<11}{'catches wrong':<16}{'escalates right':<18}")
    sweep = []
    for t in SWEEP:
        caught = sum(1 for r in wrong if r["measured"] < t)
        needless = sum(1 for r in rows if r["correct"] and r["measured"] < t)
        sweep.append({"threshold": t, "caught": caught, "needless": needless})
        mark = "  <- current" if t == get_settings().verifier_confidence_threshold else ""
        print(f"{t:<11}{f'{caught}/{len(wrong)}':<16}{f'{needless}/{len(rows) - len(wrong)}':<18}{mark}")

    if wrong:
        print("\nwrong verdicts, by measured confidence:")
        for r in sorted(wrong, key=lambda r: r["measured"]):
            want = "grounded" if r["expected"] else "ungrounded"
            print(f"  {r['measured']:.3f}  ({r['n_passages']}p, want {want})  {r['answer'][:56]}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"rows": rows, "sweep": sweep}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
