"""Knowledge-Gap — CLAUDE.md §4. Out of band, reads `audit_log` only.

Finds the questions people keep asking that the corpus can't answer, and names the
missing topic. Embeds the query text of recent refusals, clusters them by cosine
similarity (greedy, no training), and has the model summarise each cluster.

Only refusals for lack of evidence count. A `permission_conflict` refusal is not a
gap — the answer exists, the caller just can't see it — and listing those topics
would turn this report into a map of what restricted material exists.
"""

import math
from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel

from src.agents.audit import _connect

GAP_REASONS = ("insufficient_evidence", "unsupported", "low_confidence")

PROMPT_TEMPLATE = """These questions from employees went unanswered because no internal documentation covers them.

{questions}

Name the missing documentation topic in one short phrase (at most 8 words). Reply with the phrase only."""

RECENT_REFUSALS = """
    SELECT q.payload->>'query'
    FROM audit_log AS e
    JOIN audit_log AS q ON q.request_id = e.request_id AND q.event_type = 'query_received'
    WHERE e.event_type = 'escalation'
      AND e.payload->>'reason' = ANY(%(reasons)s)
      AND e.created_at >= %(since)s
    ORDER BY e.id
"""


class Gap(BaseModel):
    topic: str
    count: int
    example_queries: list[str]


class GapReport(BaseModel):
    since: datetime
    refusals_scanned: int
    gaps: list[Gap]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def cluster(vectors: list[list[float]], threshold: float) -> list[list[int]]:
    """Greedy single pass: join the first cluster whose seed is similar enough.

    ponytail: O(n·k) against each cluster's first member, order-dependent. Fine for
    a few hundred refusals a week; use HDBSCAN if it has to handle thousands.
    """
    clusters: list[list[int]] = []
    for i, v in enumerate(vectors):
        for members in clusters:
            if _cosine(v, vectors[members[0]]) >= threshold:
                members.append(i)
                break
        else:
            clusters.append([i])
    return clusters


def run_knowledge_gap_scan(
    since: datetime,
    *,
    connect: Callable = _connect,
    embeddings=None,
    chat_model=None,
    threshold: float | None = None,
) -> GapReport:
    from src.config import get_settings
    from src.llm.factory import get_chat_model, get_embeddings

    embeddings = embeddings or get_embeddings()
    chat_model = chat_model or get_chat_model()
    threshold = threshold if threshold is not None else get_settings().knowledge_gap_similarity

    with connect() as conn:
        rows = conn.execute(RECENT_REFUSALS, {"reasons": list(GAP_REASONS), "since": since})
        queries = [q for (q,) in rows if q]
    if not queries:
        return GapReport(since=since, refusals_scanned=0, gaps=[])

    vectors = embeddings.embed_documents(queries)
    gaps = []
    for members in cluster(vectors, threshold):
        asked = [queries[i] for i in members]
        prompt = PROMPT_TEMPLATE.format(questions="\n".join(f"- {q}" for q in asked[:10]))
        topic = str(getattr(chat_model.invoke(prompt), "content", "")).strip().strip('"')
        gaps.append(Gap(topic=topic, count=len(asked), example_queries=asked[:3]))
    gaps.sort(key=lambda g: g.count, reverse=True)
    return GapReport(since=since, refusals_scanned=len(queries), gaps=gaps)
