"""Connector registry, and the query-time source check (CLAUDE.md §1).

`check_access` runs at ingest time to compute the `acl_tags` written to
`document_chunks`. Those tags, and the `permissions` rows beside them, are a MIRROR
of the source's ACLs: accurate as of the last ingest, and stale afterwards. A grant
revoked in Confluence does not reach us until we re-ingest, and until then retrieval
will happily return the chunk — the staleness window every mirrored-ACL product has.

`source_grants_access` closes that window by asking the source itself, at query time.
It is deliberately applied to RESTRICTED items only: the leak that matters is the one
where the mirror still says yes about material the source now protects, and paying a
source round-trip on every internal chunk would buy little and cost latency on every
request.

Asking the source puts the source in the query path, so this fails CLOSED — an
unknown platform, a missing item, or a connector that raises all deny. A source that
is down therefore degrades to "no restricted answers", never to "restricted answers
we can no longer justify".
"""

from functools import lru_cache

from src.connectors.base import SourceConnector, SourceItem
from src.graph.state import UserContext


@lru_cache(maxsize=1)
def connectors() -> dict[str, SourceConnector]:
    """Every source connector, keyed by platform."""
    from src.connectors.confluence import ConfluenceConnector
    from src.connectors.drive import DriveConnector
    from src.connectors.jira import JiraConnector
    from src.connectors.slack import SlackConnector

    return {
        c.platform: c
        for c in (ConfluenceConnector(), JiraConnector(), SlackConnector(), DriveConnector())
    }


def source_item(platform: str, source_ref: str) -> SourceItem | None:
    """The live item behind a document, or None if the source no longer has it.

    ponytail: a linear scan of list_items() per lookup, which is nothing against four
    mock connectors. A real connector should fetch one item by ref instead.
    """
    connector = connectors().get(platform)
    if connector is None:
        return None
    return next((i for i in connector.list_items() if i.source_ref == source_ref), None)


def source_denies(user: UserContext, platform: str, source_ref: str) -> bool:
    """Should this caller be denied this document on the SOURCE's current answer?

    One source lookup, four outcomes:

    - no connector owns the platform -> False. `internal` documents are authored in
      this system and have no upstream to ask, so the mirror is already authoritative.
      Denying them here would silently empty the corpus of everything home-grown.
    - the owning connector no longer lists the item -> True. We cannot verify it, and
      a document deleted upstream should stop being answerable rather than linger.
    - listed and not restricted -> False. Internal material is left to the mirror,
      which is the point of scoping this: no source round-trip on the common path.
    - listed and restricted -> whatever `check_access` says right now.

    Anything raising on the way denies, so a broken source cannot open one.
    """
    try:
        connector = connectors().get(platform)
        if connector is None:
            return False
        item = source_item(platform, source_ref)
        if item is None:
            return True
        if item.sensitivity != "restricted":
            return False
        return not connector.check_access(user, item)
    except Exception:
        return True
