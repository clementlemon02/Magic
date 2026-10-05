"""Reconcile the mirror with every source, now.

    .venv/bin/python -m scripts.sync_sources [platform ...]

The same run the server makes on its schedule and the Sources page's "Sync now" makes. Needs
Ollama only when something changed: an unchanged document is never re-embedded. Exits 1 if any
source failed, and says which.
"""

import sys

from src.connectors import connectors
from src.db.migrate import migrate
from src.ingestion.sync import sync_sources


def main(argv: list[str]) -> int:
    known = sorted(connectors())
    unknown = [a for a in argv if a not in known]
    if unknown:
        print(f"unknown source: {', '.join(unknown)} (known: {', '.join(known)})")
        return 2
    migrate()
    reports = sync_sources(platforms=argv or None, trigger="manual")
    for r in reports:
        took = (r.finished_at - r.started_at).total_seconds()
        if r.skipped:
            print(f"{r.platform:<11} skipped: another sync of it is already running")
        elif r.error:
            print(f"{r.platform:<11} FAILED   {r.error}")
        else:
            print(
                f"{r.platform:<11} +{r.added} added  ~{r.updated} updated  -{r.removed} removed  "
                f"={r.unchanged} unchanged   grants +{r.grants_added} -{r.grants_revoked}   {took:.1f}s"
            )
    return 1 if any(r.error for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
