"""Suite-wide defaults, so unit tests stay unit tests.

Constant-time refusal is on by default and holds every refusal for seconds. Tests
that exercise it inject a clock and sleeper (test_refusal_padding.py); everything
else opts out, or the suite would sleep for real on each refusal it produces.

The answer cache is off for the same reason. create_app builds a real
PermissionAwareCache by default, and every /query fingerprints the caller against
Postgres — so from #11 until this, the API tests silently required a running
database. Tests that exercise the cache inject one explicitly.

The distilled Router is off too: it embeds every question through Ollama, which would
make each Router test depend on a running model. test_router_student.py injects its
own embeddings.
"""

import os
import tempfile

os.environ.setdefault("REFUSAL_PADDING_ENABLED", "false")
os.environ.setdefault("QUERY_CACHE_ENABLED", "false")
os.environ.setdefault("ROUTER_STUDENT_ENABLED", "false")

# Startup warm-up is off for the same reason: create_app runs in most test modules,
# and each one would otherwise load a 7B model to answer nothing.
os.environ.setdefault("WARM_ON_STARTUP", "false")

# No background sync and no startup migration: both reach for a database, and create_app
# runs in most test modules. test_sync*.py call sync_sources directly.
os.environ.setdefault("SYNC_INTERVAL_MINUTES", "0")
os.environ.setdefault("MIGRATE_ON_STARTUP", "false")

# The mock sources' authored changes live in a file (scripts/mock_source.py). Point the suite
# at one that does not exist, or a developer's leftover edit would change what every
# connector test sees. Tests of the overlay itself set their own path.
os.environ.setdefault(
    "MOCK_SOURCES_PATH", os.path.join(tempfile.mkdtemp(prefix="ib-tests-"), "mock_sources.json")
)

# A fixed signing key, so tests/helpers.py mints tokens the app under test accepts
# whatever a developer happens to have in their .env.
os.environ.setdefault("AUTH_SECRET", "test-only-secret")


import pytest  # noqa: E402 - after the environment defaults above, which must be set first


@pytest.fixture(autouse=True)
def no_freshness_lookup(monkeypatch):
    """An answer's citations are stamped with when their source last synced, which reads
    the database. Unit tests stay unit tests; the ones about the stamp inject their own."""
    monkeypatch.setattr("src.api.main.last_synced", lambda: {})
