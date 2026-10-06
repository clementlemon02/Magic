"""Play the part of an author in the mock source systems, so a sync has something to find.

    .venv/bin/python -m scripts.mock_source list
    .venv/bin/python -m scripts.mock_source edit confluence:SUPPORT/refund-policy "45 days" "30 days"
    .venv/bin/python -m scripts.mock_source access confluence:SUPPORT/refund-policy compliance
    .venv/bin/python -m scripts.mock_source delete jira:PAY/ENG-4471
    .venv/bin/python -m scripts.mock_source restore jira:PAY/ENG-4471     # drop one change
    .venv/bin/python -m scripts.mock_source reset                         # drop all of them

Nothing here touches the database. It changes what the sources SAY (src/connectors/overlay.py),
exactly as an edit in Confluence would; the mirror catches up on the next sync, or at once with
`python -m scripts.sync_sources`. Restricted documents are the exception: the query-time recheck
asks the source directly, so narrowing one takes effect before any sync.

`access` sets who may read an item: Confluence groups, Drive ACL entries, a Jira role, or Slack
member ids (which also makes the channel private).
"""

import argparse
import sys
from datetime import UTC, datetime

from src.connectors import connectors, overlay, source_item
from src.connectors.base import (
    ConfluencePermission,
    DrivePermission,
    JiraPermission,
    SlackPermission,
)


def _locate(ref: str):
    platform, _, source_ref = ref.partition(":")
    item = source_item(platform, source_ref)
    if item is None:
        sys.exit(f"no such item: {ref}  (try `list`)")
    return connectors()[platform], item


def _with_access(permission, values: list[str]):
    if isinstance(permission, ConfluencePermission):
        return permission.model_copy(update={"viewer_groups": values})
    if isinstance(permission, DrivePermission):
        return permission.model_copy(update={"acl_entries": values})
    if isinstance(permission, JiraPermission):
        return permission.model_copy(update={"role_required": values[0]})
    if isinstance(permission, SlackPermission):
        return permission.model_copy(
            update={"is_private": True, "member_ids": [int(v) for v in values]}
        )
    raise TypeError(type(permission))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("reset")
    for name in ("delete", "restore"):
        commands.add_parser(name).add_argument("ref")
    edit = commands.add_parser("edit")
    edit.add_argument("ref")
    edit.add_argument("find")
    edit.add_argument("replacement")
    access = commands.add_parser("access")
    access.add_argument("ref")
    access.add_argument("who", help="comma-separated")
    args = parser.parse_args(argv)

    if args.command == "list":
        changed = overlay.authored()
        for platform, connector in connectors().items():
            for item in connector.list_items():
                mark = "*" if f"{platform}:{item.source_ref}" in changed else " "
                print(f"{mark} {platform}:{item.source_ref:<34} {item.sensitivity:<10} {item.title}")
        for key, entry in changed.items():
            if entry["op"] == "delete":
                print(f"* {key}  (deleted)")
        return 0
    if args.command == "reset":
        overlay.reset()
        print("sources are back to their baseline")
        return 0
    if args.command == "restore":
        platform, _, source_ref = args.ref.partition(":")
        print("restored" if overlay.restore(platform, source_ref) else "nothing to restore")
        return 0
    if args.command == "delete":
        platform, _, source_ref = args.ref.partition(":")
        _locate(args.ref)
        overlay.delete_item(platform, source_ref)
        print(f"deleted {args.ref} at the source")
        return 0

    connector, item = _locate(args.ref)
    if args.command == "edit":
        if args.find not in item.content:
            sys.exit(f"{args.find!r} is not in that item's text")
        item = item.model_copy(update={
            "content": item.content.replace(args.find, args.replacement),
            "updated_at": datetime.now(UTC),
        })
        overlay.set_item(item, connector.permissions_for(item))
        print(f"edited {args.ref}")
    else:
        item = item.model_copy(update={"updated_at": datetime.now(UTC)})
        permission = _with_access(connector.permissions_for(item), args.who.split(","))
        overlay.set_item(item, permission)
        print(f"{args.ref} is now readable by: {args.who}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
