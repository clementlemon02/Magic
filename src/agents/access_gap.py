"""Access-Gap report. Out of band, reads `audit_log` only.

The other half of the Knowledge-Gap report. That one finds questions no document
answers — write the missing page. This one finds questions a document DOES answer
that the asker could not see:

    knowledge gap -> the answer does not exist       -> write documentation
    access gap    -> the answer exists, behind a wall -> fix the permission model

Both signals are already on the record, so this only has to count them:

- `permission_conflict` — the mirror refused: a restricted document outranked
  everything the caller could see.
- `source_recheck_denied` — the source refused at query time, after our mirror had
  already allowed it. A document showing up here is a document whose ACLs drifted.

Ranked by how many DIFFERENT people hit the same wall, not by total requests: one
person retrying is a bad afternoon, six people from four teams is a permission model
that does not match how the company actually works.

SECURITY: unlike the knowledge-gap report, this deliberately names restricted items,
so it is a map of what restricted material exists and who wanted it. It is reachable
only from the compliance-officer routes, and must never be exposed to an asker or
folded into anything that is. Same reasoning as `knowledge_gap.py`, opposite
conclusion, because the audience is different.
"""

from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel

from src.agents.audit import _connect
from src.graph.state import SourcePlatform

# One row per (request, item) that someone was refused, with the question they asked.
# The two signals are unioned so a document that drifts AND outranks is counted once
# per request rather than once per detector.
REFUSED_ITEMS = """
    WITH refused AS (
        SELECT e.request_id, e.user_id, e.created_at,
               item->>'source_platform' AS source_platform,
               item->>'source_ref'      AS source_ref,
               item->>'sensitivity'     AS sensitivity,
               'permission_conflict'    AS signal
        FROM audit_log AS e
        CROSS JOIN LATERAL jsonb_array_elements(e.payload->'conflicts') AS item
        WHERE e.event_type = 'permission_conflict' AND e.created_at >= %(since)s

        UNION ALL

        SELECT e.request_id, e.user_id, e.created_at,
               item->>'source_platform' AS source_platform,
               item->>'source_ref'      AS source_ref,
               NULL                     AS sensitivity,
               'source_recheck_denied'  AS signal
        FROM audit_log AS e
        CROSS JOIN LATERAL jsonb_array_elements(e.payload->'items') AS item
        WHERE e.event_type = 'source_recheck_denied' AND e.created_at >= %(since)s
    )
    SELECT refused.source_platform, refused.source_ref, refused.sensitivity,
           refused.signal, refused.request_id, refused.user_id,
           question.payload->>'query' AS query
    FROM refused
    LEFT JOIN audit_log AS question
      ON question.request_id = refused.request_id
     AND question.event_type = 'query_received'
    -- Most recent first, so `example_queries` is who hit this wall latest, and is
    -- the same list on a re-run. request_id is a random UUID and orders arbitrarily.
    ORDER BY refused.created_at DESC, refused.request_id
"""


class AccessGap(BaseModel):
    source_platform: SourcePlatform
    source_ref: str
    sensitivity: str | None
    signals: list[str]
    requests: int
    askers: int
    example_queries: list[str]


class AccessGapReport(BaseModel):
    since: datetime
    refusals_scanned: int
    gaps: list[AccessGap]


def run_access_gap_scan(since: datetime, *, connect: Callable = _connect) -> AccessGapReport:
    """Rank the documents people were refused, by how many distinct people asked."""
    with connect() as conn:
        rows = list(conn.execute(REFUSED_ITEMS, {"since": since}))

    grouped: dict[tuple[str, str], dict] = {}
    for platform, ref, sensitivity, signal, request_id, user_id, query in rows:
        gap = grouped.setdefault(
            (platform, ref),
            {"sensitivity": None, "signals": set(), "requests": set(), "askers": set(),
             "queries": []},
        )
        gap["sensitivity"] = gap["sensitivity"] or sensitivity
        gap["signals"].add(signal)
        gap["requests"].add(request_id)
        if user_id is not None:
            gap["askers"].add(user_id)
        if query and query not in gap["queries"]:
            gap["queries"].append(query)

    gaps = [
        AccessGap(
            source_platform=platform,
            source_ref=ref,
            sensitivity=gap["sensitivity"],
            signals=sorted(gap["signals"]),
            requests=len(gap["requests"]),
            askers=len(gap["askers"]),
            example_queries=gap["queries"][:3],
        )
        for (platform, ref), gap in grouped.items()
    ]
    # Distinct people first: that is the signal the permission model is wrong.
    gaps.sort(key=lambda g: (g.askers, g.requests), reverse=True)
    return AccessGapReport(
        since=since, refusals_scanned=len({r[4] for r in rows}), gaps=gaps
    )
