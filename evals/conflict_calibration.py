"""Does the permission-conflict check fire on the right questions? Measured on the seeded corpus.

    .venv/bin/python -m scripts.seed_demo && .venv/bin/python -m evals.conflict_calibration

A conflict means "a restricted document outranked everything this caller may see",
and it is what the compliance officer reads in the explanation. Firing on an unrelated
question tells them restricted material was withheld when it wasn't. Missing a related
one records the refusal under the wrong reason and hides it from the conflict report.

Runs Retrieval's own `retrieve()` for Alex (support, clearance 0) against the live
database, so it measures the real decision, floor and margin included. Re-run it
whenever the corpus or the embedding model changes: on the original eight
one-sentence documents, "What is the office wifi password?" was reported as
withholding the AML evidence register.
"""

import json
import sys
from pathlib import Path

from src.agents.retrieval import _psycopg_execute, retrieve
from src.api.main import load_user
from src.config import get_settings
from src.llm.factory import get_embeddings

# Questions whose best answer is restricted. Each should raise a conflict for Alex.
ABOUT_RESTRICTED = [
    "What triggers an AML escalation review?",
    "Which accounts are under Tier 2 review?",
    "What is the AML threshold for transfers to high-risk jurisdictions?",
    "What did the TM-7 lookback find?",
    "What is in the AML evidence register?",
    "What is Project Nightingale?",
    "Is account 7731 being investigated?",
    "How long do analysts have to file a suspicious transaction report?",
    "Which AML cases are open?",
]

# Questions nothing restricted answers. None should raise a conflict.
UNRELATED = [
    "What is the office wifi password?",
    "Can fathers take paternity leave?",
    "What is our parental leave policy?",
    "How do I get a replacement laptop?",
    "Do we reimburse gym memberships?",
    "Is there a dress code for client meetings?",
    "Which VPN should contractors use?",
    "What is our position on cryptocurrency custody?",
    "How many vacation days carry over into next year?",
    "Who approves overtime for weekend shifts?",
    "Where do I submit a travel expense claim?",
    "How long do customers have to contest a chargeback?",
    "What caused the payment outage?",
]

RESULTS = Path(__file__).parent / "results" / "conflict_calibration.json"


def main() -> int:
    settings = get_settings()
    alex = load_user(1)
    embeddings = get_embeddings()

    def conflicts_for(query: str) -> list[str]:
        _, conflicts = retrieve(
            query, alex, _psycopg_execute, embeddings,
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
            conflict_score_margin=settings.permission_conflict_score_margin,
        )
        return [f"{c.source_platform}:{c.source_ref}" for c in conflicts]

    rows = [(q, True, conflicts_for(q)) for q in ABOUT_RESTRICTED]
    rows += [(q, False, conflicts_for(q)) for q in UNRELATED]

    caught = sum(1 for _, want, got in rows if want and got)
    false_alarms = [(q, got) for q, want, got in rows if not want and got]
    missed = [q for q, want, got in rows if want and not got]

    print(f"floor {settings.retrieval_min_score} · margin {settings.permission_conflict_score_margin}")
    print(f"conflicts raised on restricted questions  {caught}/{len(ABOUT_RESTRICTED)}")
    print(f"false conflicts on unrelated questions    {len(false_alarms)}/{len(UNRELATED)}")
    for q, got in false_alarms:
        print(f"  FALSE  {q}  ->  {', '.join(got)}")
    for q in missed:
        # Still refused, via the no-evidence path, and logged under that reason.
        print(f"  missed {q}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "min_score": settings.retrieval_min_score,
        "margin": settings.permission_conflict_score_margin,
        "rows": [{"query": q, "should_conflict": w, "conflicts": g} for q, w, g in rows],
    }, indent=2))
    # A false conflict misinforms the auditor; that is the failure this guards.
    return 1 if false_alarms else 0


if __name__ == "__main__":
    sys.exit(main())
