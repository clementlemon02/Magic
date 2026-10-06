"""Sweep RETRIEVAL_MIN_SCORE: does raising the floor ever cost real recall, for ANY persona?

    LLM_BACKEND=ollama .venv/bin/python -m evals.retrieval_threshold_sweep

Raising the floor stops off-topic chunks being read by the model (prefill dominates
Synthesizer and Verifier latency), so it is tempting. The first version of this sweep
asked every question as one support user, found recall held to 0.68, and recommended
0.65. That broke the compliance officer: the AML documents score only 0.53-0.59
against their own questions, so at 0.65 they were refused outright, and restricted
matches stopped registering as permission conflicts (conflict recall 7/9 -> 5/9).

A floor is one number applied to every persona's region of the corpus, and similarity
runs lower in some regions than others, so this sweep asks as each persona.
`evals/conflict_calibration.py` measures the OTHER search this constant gates; a
threshold change needs both green.

Per CLAUDE.md's convention for this constant ("per-model; see .env.example, re-measure
if the embedding model changes") the value is Clement's call. This exists so that call
is made from numbers.
"""

import sys

import psycopg

from src.agents.retrieval import retrieve
from src.config import get_settings
from src.graph.state import UserContext
from src.llm.factory import get_embeddings

SUPPORT = UserContext(id=1, role="support", dept="support", clearance_level=0)
COMPLIANCE = UserContext(id=2, role="compliance", dept="compliance", clearance_level=1)

# (asker, question, (platform, source_ref) pairs that MUST still be retrieved). Identified
# by citation, not by a numeric document_id: the id is whatever order seed_demo.py
# happened to insert rows in, which is not the fixture numbering evals/cases.py uses.
# None means "must retrieve nothing" - the unanswerable case, where a floor set too LOW
# is the failure (noise masquerading as evidence), not too high.
CASES: list[tuple[UserContext, str, set[tuple[str, str]] | None]] = [
    (SUPPORT, "How long do customers have to contest a chargeback?",
     {("confluence", "SUPPORT/refund-policy")}),
    (SUPPORT, "What caused the payment outage?",
     {("jira", "PAY/ENG-4471")}),
    (SUPPORT, "How is customer PII handled, and how does on-call get access?",
     {("drive", "file-pii-standard"), ("slack", "support-eng/1726000000.000100")}),
    (SUPPORT, "What is our parental leave policy?", None),
    (SUPPORT, "How many employees does the company have?", None),
    (SUPPORT, "What is our stance on cryptocurrency custody?", None),
    # The compliance officer's own questions. Their best matches score 0.53-0.79, well
    # below the support cases' 0.69-0.77, which is exactly what a single floor must survive.
    (COMPLIANCE, "What triggers an AML escalation review?",
     {("confluence", "COMPLIANCE/aml-escalation")}),
    (COMPLIANCE, "What does the AML evidence register contain?",
     {("drive", "file-aml-evidence")}),
]

CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75]


def _execute(sql, params):
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _who(user: UserContext) -> str:
    return user.role


def main() -> int:
    settings = get_settings()
    embeddings = get_embeddings()

    # Retrieve once per (asker, question) at min_score=0.0, then re-threshold in Python.
    raw: dict[tuple[int, str], list] = {}
    for user, question, _ in CASES:
        chunks, _ = retrieve(
            question, user, _execute, embeddings,
            top_k=6, min_score=0.0, conflict_score_margin=settings.permission_conflict_score_margin,
        )
        raw[(user.id, question)] = chunks

    print(f"backend embeddings: {settings.ollama_embedding_model}\n")
    header = f"{'floor':>6}  {'recall':>13}  {'chunks kept':>12}  {'noise kept':>11}  lost recall for"
    print(header)
    print("-" * (len(header) + 24))

    first_loss: dict[str, float] = {}
    for floor in CANDIDATES:
        kept_total = noise = 0
        lost: list[str] = []
        for user, question, required in CASES:
            kept = [c for c in raw[(user.id, question)] if c.score >= floor]
            refs = {(c.citation.source_platform, c.citation.source_ref) for c in kept}
            kept_total += len(kept)
            if required is not None:
                noise += len(refs - required)
                if not required.issubset(refs):
                    lost.append(f"{_who(user)}: {question[:34]}")
                    first_loss.setdefault(f"{_who(user)}: {question}", floor)
            else:
                noise += len(kept)
        status = "OK" if not lost else "LOSES RECALL"
        print(f"{floor:>6.2f}  {status:>13}  {kept_total:>12}  {noise:>11}  {'; '.join(lost)}")

    print(f"\nscores at the current setting (RETRIEVAL_MIN_SCORE={settings.retrieval_min_score}):")
    for user, question, required in CASES:
        print(f"  [{_who(user)}] {question}")
        for c in raw[(user.id, question)]:
            ref = (c.citation.source_platform, c.citation.source_ref)
            mark = " <- required" if required and ref in required else ""
            print(f"      {c.score:.3f}  {c.citation.source_platform}:{c.citation.source_ref}{mark}")

    if first_loss:
        print("\nfirst floor that loses each required document:")
        for who_q, floor in first_loss.items():
            print(f"  {floor:.2f}  {who_q}")
    print("\nAlso run evals/conflict_calibration.py: this constant gates the conflict search too.")
    print("This is Clement's call - CLAUDE.md's own convention for this constant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
