"""Does a semantic answer cache leak across permission boundaries? Measured.

    LLM_BACKEND=ollama .venv/bin/python -m evals.cache_probe

Runs the same three probes against a plain semantic cache and the permission-aware
one, using real fingerprints read from the live `permissions` table. The naive cache
is a fair implementation of the ordinary technique — embed the query, reuse the
answer for a similar one — not a strawman. That is exactly what makes it leak.
"""

import sys

import psycopg

from src.cache import PermissionAwareCache, _cosine
from src.config import get_settings
from src.graph.state import AskerResponse, UserContext
from src.llm.factory import get_embeddings

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)
MARCUS = UserContext(id=2, role="compliance", dept="compliance", clearance_level=1)

RESTRICTED_ANSWER = AskerResponse(
    text="AML escalation is triggered for transfers at or above SGD 9,500 to high-risk "
    "jurisdictions, which start a Tier 2 review."
)
QUESTION = "What triggers an AML escalation review?"
CANARIES = ("9,500", "tier 2", "high-risk")


class NaiveSemanticCache:
    """The ordinary technique: key on the query, ignore who is asking."""

    def __init__(self, embeddings, threshold: float = 0.93):
        self._embeddings = embeddings
        self.threshold = threshold
        self._entries: list[tuple[list[float], AskerResponse]] = []

    def get(self, query: str, user: UserContext) -> AskerResponse | None:
        vector = self._embeddings.embed_query(query)
        for stored, response in self._entries:
            if _cosine(vector, stored) >= self.threshold:
                return response
        return None

    def put(self, query: str, user: UserContext, response: AskerResponse) -> None:
        self._entries.append((self._embeddings.embed_query(query), response))


def _set_revoked(revoked: bool) -> None:
    clause = "now()" if revoked else "NULL"
    with psycopg.connect(get_settings().database_url) as conn:
        conn.execute(
            f"UPDATE permissions SET revoked_at = {clause} "
            "WHERE user_id = 2 AND source_ref = 'COMPLIANCE/aml-escalation'"
        )
        conn.commit()


def _leaked(response: AskerResponse | None) -> bool:
    if response is None:
        return False
    lowered = response.text.lower()
    return any(canary in lowered for canary in CANARIES)


def main() -> int:
    embeddings = get_embeddings()
    naive = NaiveSemanticCache(embeddings)
    aware = PermissionAwareCache(embeddings=embeddings, similarity_threshold=0.93)

    print(f"backend: {get_settings().llm_backend} · embeddings: {get_settings().ollama_embedding_model}")
    print("Marcus (compliance) asks a restricted question; his answer is cached.\n")

    _set_revoked(False)
    for cache in (naive, aware):
        cache.put(QUESTION, MARCUS, RESTRICTED_ANSWER)

    rows = []

    # 1. A lower-privileged caller asks the same question.
    rows.append(("Alex (support) asks the same question",
                 _leaked(naive.get(QUESTION, ALEX)),
                 _leaked(aware.get(QUESTION, ALEX))))

    # 2. A paraphrase, which is the whole point of a semantic cache.
    paraphrase = "Tell me when an AML escalation review gets triggered"
    rows.append(("Alex asks a paraphrase of it",
                 _leaked(naive.get(paraphrase, ALEX)),
                 _leaked(aware.get(paraphrase, ALEX))))

    # 3. Marcus keeps his own access, so his own entry should still serve him.
    marcus_ok_naive = naive.get(QUESTION, MARCUS) is not None
    marcus_ok_aware = aware.get(QUESTION, MARCUS) is not None

    # 4. Revoke Marcus's grant mid-demo (Scenario 4) and ask again.
    _set_revoked(True)
    after_naive = _leaked(naive.get(QUESTION, MARCUS))
    after_aware = _leaked(aware.get(QUESTION, MARCUS))
    _set_revoked(False)
    rows.append(("Marcus asks after his access is revoked", after_naive, after_aware))

    print(f"{'probe':<44}{'naive':>10}{'ours':>10}")
    print("-" * 64)
    for label, naive_leak, aware_leak in rows:
        print(f"{label:<44}{('LEAK' if naive_leak else 'clean'):>10}{('LEAK' if aware_leak else 'clean'):>10}")
    print(f"\n{'still serves its owner while permitted':<44}"
          f"{('yes' if marcus_ok_naive else 'no'):>10}{('yes' if marcus_ok_aware else 'no'):>10}")

    # The other half of the case: what a hit actually saves.
    import time

    started = time.monotonic()
    for _ in range(5):
        aware.get(QUESTION, MARCUS)
    hit_cost = (time.monotonic() - started) / 5
    print(f"\ncache hit costs {hit_cost:.2f}s (one embedding + one indexed query)")
    print("a miss runs retrieval plus three model calls — 4.7s median, measured in evals/run.py")

    leaks = sum(1 for _, n, _ in rows if n), sum(1 for _, _, a in rows if a)
    print(f"\nleaks: naive {leaks[0]}/3 · permission-aware {leaks[1]}/3")
    print("The permission-aware cache keeps the benefit: same access, same entry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
