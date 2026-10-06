"""What authors do in the source systems, for the mock connectors.

A real connector asks Confluence, Jira, Slack or Drive what is there NOW. The mocks keep
their baseline content in code, so on their own they could never change, and a sync built
against them could never be shown to work. This is the mock source's own state: a small
JSON file that one process (scripts/mock_source.py) writes and every connector reads, so
an edit made while the server runs is exactly as visible to the next sync as an edit in
the real platform would be.

The file is keyed by "<platform>:<source_ref>":

    {"confluence:SUPPORT/refund-policy": {"op": "set", "item": {...SourceItem...},
                                          "permission": {...native permission...}},
     "jira:PAY/ENG-4471": {"op": "delete"}}

`set` replaces the baseline item whole, or adds one that has no baseline. `delete` hides
it. Nothing here is consulted for authorization: `permission` is the SOURCE's answer to
"who may read this", which ingestion mirrors into acl_tags and grants like any other.
"""

import json
import os
import tempfile
from pathlib import Path

from pydantic import TypeAdapter

from src.connectors.base import NativePermission, SourceItem

_PERMISSION = TypeAdapter(NativePermission)
_REPO_ROOT = Path(__file__).resolve().parents[2]

# (path, mtime_ns, size) -> parsed file. The query-time source recheck lists a connector's
# items once per restricted chunk, so reading and parsing the file on every call would be a
# cost the request path has no business paying; a stat is cheap and always current.
_cache: tuple[tuple[Path, int, int], dict] | None = None


def path() -> Path:
    from src.config import get_settings

    configured = Path(get_settings().mock_sources_path)
    return configured if configured.is_absolute() else _REPO_ROOT / configured


def _read() -> dict:
    global _cache
    target = path()
    try:
        stat = target.stat()
    except FileNotFoundError:
        _cache = None
        return {}
    stamp = (target, stat.st_mtime_ns, stat.st_size)
    if _cache is not None and _cache[0] == stamp:
        return _cache[1]
    data = json.loads(target.read_text(encoding="utf-8") or "{}")
    _cache = (stamp, data)
    return data


def _write(entries: dict) -> None:
    """Atomic: a reader mid-write must see the old file or the new one, never half."""
    global _cache
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2, default=str)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    _cache = None


def items(platform: str, baseline: list[SourceItem]) -> list[SourceItem]:
    """The platform's items as the source would list them now: baseline plus authored changes."""
    entries = _read()
    prefix = f"{platform}:"
    merged = {i.source_ref: i for i in baseline}
    for key, entry in entries.items():
        if not key.startswith(prefix):
            continue
        ref = key[len(prefix):]
        if entry["op"] == "delete":
            merged.pop(ref, None)
        else:
            merged[ref] = SourceItem.model_validate(entry["item"])
    return list(merged.values())


def permission(platform: str, source_ref: str) -> NativePermission | None:
    """The authored permission for an item, or None to fall back to the baseline's."""
    entry = _read().get(f"{platform}:{source_ref}")
    if not entry or entry["op"] != "set" or not entry.get("permission"):
        return None
    return _PERMISSION.validate_python(entry["permission"])


def set_item(item: SourceItem, permission: NativePermission | None = None) -> None:
    entries = dict(_read())
    entry: dict = {"op": "set", "item": json.loads(item.model_dump_json())}
    if permission is not None:
        entry["permission"] = json.loads(permission.model_dump_json())
    entries[f"{item.platform}:{item.source_ref}"] = entry
    _write(entries)


def delete_item(platform: str, source_ref: str) -> None:
    entries = dict(_read())
    entries[f"{platform}:{source_ref}"] = {"op": "delete"}
    _write(entries)


def restore(platform: str, source_ref: str) -> bool:
    """Drop the authored change, so the item is its baseline again (or gone, if it had none)."""
    entries = dict(_read())
    removed = entries.pop(f"{platform}:{source_ref}", None) is not None
    if removed:
        _write(entries)
    return removed


def reset() -> None:
    """Every authored change gone: the sources are back to their baseline."""
    target = path()
    if target.exists():
        target.unlink()
    global _cache
    _cache = None


def authored() -> dict:
    """What has been changed, for a human to read."""
    return dict(_read())
