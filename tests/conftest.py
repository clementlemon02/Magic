"""Suite-wide defaults, so unit tests stay unit tests.

Constant-time refusal is on by default and holds every refusal for seconds. Tests
that exercise it inject a clock and sleeper (test_refusal_padding.py); everything
else opts out, or the suite would sleep for real on each refusal it produces.

The answer cache is off for the same reason. create_app builds a real
PermissionAwareCache by default, and every /query fingerprints the caller against
Postgres — so from #11 until this, the API tests silently required a running
database. Tests that exercise the cache inject one explicitly.
"""

import os

os.environ.setdefault("REFUSAL_PADDING_ENABLED", "false")
os.environ.setdefault("QUERY_CACHE_ENABLED", "false")
