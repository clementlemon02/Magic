"""Sweep RETRIEVAL_MIN_SCORE: how much prefill is wasted on off-topic chunks, and
does raising the floor ever cost real recall?

Found while chasing a live complaint about latency: prefill dominates both the
Synthesizer and Verifier calls (measured: ~1.3-1.6s of a ~2-2.7s call is reading the
prompt, not writing the answer — see PR discussion). Retrieval was handing the
Synthesizer chunks that scored as low as 0.63 on a query about chargeback windows,
where the actual answer sat in two passages at 0.75 and 0.77 — three off-topic,
"vaguely support-adjacent" passages were read and paid for in full, then ignored.

    LLM_BACKEND=ollama .venv/bin/python -m evals.retrieval_threshold_sweep

Per CLAUDE.md's own convention for this constant ("per-model; see .env.example,
re-measure if the embedding model changes") — this is Clement's call, not mine.
This sweep exists so that call is made from numbers.
"""

import sys

import psycopg

from src.agents.retrieval import retrieve
from src.config import get_settings
from src.graph.state import UserContext
from src.llm.factory import get_embeddings

SUPPORT = UserContext(id=1, role="support", dept="support", clearance_level=0)

# (question, (platform, source_ref) pairs that MUST still be retrieved). Identified
# by citation, not by a numeric document_id: the id is whatever order seed_demo.py
# happened to insert rows in, which is not the fixture numbering evals/cases.py uses
# for its own synthetic corpus — copying THOSE numbers over here silently checked
# recall against the wrong documents entirely (caught by this comment existing).
# None means "must retrieve nothing" — the unanswerable case, where a threshold set
# too LOW is the failure (noise masquerading as evidence), not too high.
CASES: list[tuple[str, set[tuple[str, str]] | None]] = [
    ("How long do customers have to contest a chargeback?",
     {("confluence", "SUPPORT/refund-policy")}),
    ("What caused the payment outage?",
     {("jira", "PAY/ENG-4471")}),
    ("How is customer PII handled, and how does on-call get access?",
     {("drive", "file-pii-standard"), ("slack", "support-eng/1726000000.000100")}),
    ("What is our parental leave policy?", None),
    ("How many employees does the company have?", None),
    ("What is our stance on cryptocurrency custody?", None),
]

CANDIDATES = [0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75]


def _execute(sql, params):
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def main() -> int:
    settings = get_settings()
    embeddings = get_embeddings()

    # Retrieve once per question at min_score=0.0 (everything), then re-threshold in
    # Python — one embedding call per question instead of one per (question, candidate).
    raw: dict[str, list] = {}
    for question, _ in CASES:
        chunks, _ = retrieve(
            question, SUPPORT, _execute, embeddings,
            top_k=6, min_score=0.0, conflict_score_margin=settings.permission_conflict_score_margin,
        )
        raw[question] = chunks

    print(f"backend embeddings: {settings.ollama_embedding_model}\n")
    header = f"{'threshold':>9}  {'recall':>8}  {'chunks kept':>12}  {'chunks dropped':>15}  {'wasted (noise) fetched':>22}"
    print(header)
    print("-" * len(header))

    failed_recall_at = None
    for threshold in CANDIDATES:
        kept_total = dropped_total = 0
        recall_ok = True
        noise_still_included = 0
        for question, required in CASES:
            chunks = raw[question]
            kept = [c for c in chunks if c.score >= threshold]
            kept_refs = {(c.citation.source_platform, c.citation.source_ref) for c in kept}
            dropped_total += len(chunks) - len(kept)
            kept_total += len(kept)
            if required is not None and not required.issubset(kept_refs):
                recall_ok = False
            if required is not None:
                noise_still_included += len(kept_refs - required)
            elif kept:
                # Unanswerable case: anything kept here IS noise, by definition.
                noise_still_included += len(kept)
        status = "OK" if recall_ok else "LOSES RECALL"
        print(f"{threshold:>9.2f}  {status:>8}  {kept_total:>12}  {dropped_total:>15}  {noise_still_included:>22}")
        if not recall_ok and failed_recall_at is None:
            failed_recall_at = threshold

    print("\nper-question scores at the current setting"
          f" (RETRIEVAL_MIN_SCORE={settings.retrieval_min_score}):")
    for question, required in CASES:
        print(f"  {question}")
        for c in raw[question]:
            ref = (c.citation.source_platform, c.citation.source_ref)
            required_note = " <- required" if required and ref in required else ""
            print(f"      {c.score:.3f}  {c.citation.source_platform}:{c.citation.source_ref}{required_note}")

    if failed_recall_at is not None:
        print(f"\nfirst threshold that loses required recall: {failed_recall_at}")
    print("\nThis is Clement's call — CLAUDE.md's own convention for this constant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
