"""Ask real questions as several personas, so the access-gap report has something to show.

    .venv/bin/python -m scripts.seed_activity

Runs each question through the real POST /query — router, retrieval, escalation,
audit — so every row it leaves behind was written by the graph rather than by this
script. Needs `docker compose up -d`, a seeded corpus (scripts.seed_demo) and Ollama.

APPENDS to audit_log and never truncates it: the chain is append-only by design, and
a seeding script must not become a way to rewrite history. Re-running it therefore
adds more activity rather than replacing what is there, which is the honest
behaviour even though it means the counts grow.

The asks are chosen so the report has a shape worth looking at: the AML material is
wanted by people in three different departments, while the evidence register is
wanted by one. The point of the report is the FIRST number — distinct askers.
"""

import sys

from fastapi.testclient import TestClient

from src.api.main import create_app

# (user_id, question). Personas from scripts/seed_users.sql.
ASKS = [
    (1, "What triggers an AML escalation review?"),            # Alex, support
    (3, "What is the AML threshold for high-risk transfers?"),  # Priya, support
    (4, "Which accounts are under Tier 2 review?"),             # Daniel, support
    (5, "What did the TM-7 lookback find?"),                    # Wei, engineering
    (6, "Is account 7731 being investigated?"),                 # Sofia, engineering
    (8, "What is in the AML evidence register?"),               # Raj, operations
    (3, "What is Project Nightingale?"),                        # Priya again
    (1, "How long do customers have to contest a chargeback?"),  # answerable, for contrast
]


def main() -> int:
    client = TestClient(create_app())
    if client.get("/health").status_code != 200:
        print("API is not healthy — check Docker and Ollama")
        return 1

    for index, (user_id, question) in enumerate(ASKS, start=1):
        response = client.post("/query", json={"user_id": user_id, "query": question})
        if response.status_code != 200:
            print(f"  {index}/{len(ASKS)} user {user_id}: HTTP {response.status_code}")
            continue
        # Refusals are byte-identical by design (§5), so report the shape, not the text.
        refused = response.json()["text"].startswith("I don't have an answer")
        print(f"  {index}/{len(ASKS)} user {user_id}: {'refused' if refused else 'answered'}  {question}")

    print("\nSee it at /dashboard as user 2 or 7, or GET /access-gaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
