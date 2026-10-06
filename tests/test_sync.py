"""The source sync's decisions, and the mock sources it reads. No database.

The reconcile itself runs against Postgres in test_sync_integration.py. What is tested here
is the part that has to be right whatever the database does: what counts as a change, and
which grants a sync may add and take away.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import get_settings
from src.connectors import connectors, overlay, source_denies
from src.connectors.base import ConfluencePermission, SourceItem
from src.graph.state import UserContext
from src.ingestion.sync import SOURCE, Held, SyncReport, Wanted, _wanted, grant_changes, make_plan
from tests.helpers import as_user

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)
MARCUS = UserContext(id=2, role="compliance", dept="compliance", clearance_level=1)


# --- what counts as a change -----------------------------------------------------------

def _item(ref="a", content="one two three", title="Title", dept="support", sensitivity="internal"):
    return SourceItem(
        platform="confluence", source_ref=ref, title=title, content=content,
        dept=dept, sensitivity=sensitivity,
    )


class FakeSource:
    """A connector that says whatever the test says is at the source."""

    platform = "confluence"

    def __init__(self, items, groups=("support",)):
        self._items, self._groups = items, list(groups)

    def list_items(self):
        return self._items

    def permissions_for(self, item):
        return ConfluencePermission(space="S", page=item.source_ref, viewer_groups=self._groups)


def _wanted_one(**kwargs) -> Wanted:
    groups = kwargs.pop("groups", ("support",))
    return next(iter(_wanted(FakeSource([_item(**kwargs)], groups)).values()))


def _held_like(w: Wanted, **changes) -> Held:
    fields = dict(
        id=1, source_ref=w.item.source_ref, content_hash=w.content_hash, title=w.item.title,
        dept=w.item.dept, sensitivity=w.item.sensitivity, acl_tags=list(w.acl_tags),
    )
    return Held(**{**fields, **changes})


def test_a_new_item_is_added():
    w = _wanted_one()
    plan = make_plan({"a": w}, {})
    assert plan.add == [w] and plan.changed


def test_an_untouched_item_is_left_alone():
    w = _wanted_one()
    plan = make_plan({"a": w}, {"a": _held_like(w)})
    assert plan.unchanged == 1 and not plan.changed


def test_an_edit_is_a_rewrite():
    old = _wanted_one(content="refunds take 45 days")
    new = _wanted_one(content="refunds take 30 days")
    plan = make_plan({"a": new}, {"a": _held_like(old)})
    assert [(h.id, w.content_hash) for h, w in plan.rewrite] == [(1, new.content_hash)]
    assert not plan.add and not plan.retag


def test_reflowing_a_page_is_not_an_edit():
    """Re-embedding a page because someone re-wrapped its lines would be cost for nothing,
    and would bump the cache epoch — emptying every cached answer — on a no-op."""
    reflowed = _wanted_one(content="one   two\n\n three ")
    assert reflowed.content_hash == _wanted_one(content="one two three").content_hash


def test_a_changed_acl_is_a_retag_not_a_rewrite():
    """Who may read it changed, the words did not: update in place, embed nothing."""
    before = _wanted_one(groups=("support",))
    after = _wanted_one(groups=("compliance",))
    plan = make_plan({"a": after}, {"a": _held_like(before)})
    assert len(plan.retag) == 1 and not plan.rewrite


def test_reordering_the_acl_is_not_a_change():
    before = _wanted_one(groups=("support", "all-staff"))
    after = _wanted_one(groups=("all-staff", "support"))
    assert not make_plan({"a": after}, {"a": _held_like(before)}).changed


@pytest.mark.parametrize("change", [{"title": "New title"}, {"sensitivity": "restricted"}, {"dept": "legal"}])
def test_title_department_and_sensitivity_changes_are_retags(change):
    before = _wanted_one()
    after = _wanted_one(**change)
    plan = make_plan({"a": after}, {"a": _held_like(before)})
    assert len(plan.retag) == 1


def test_an_edit_that_also_changes_the_acl_is_one_rewrite():
    """The rewrite writes the ACL too; listing it as a retag as well would apply it twice."""
    before = _wanted_one(content="old", groups=("support",))
    after = _wanted_one(content="new", groups=("compliance",))
    plan = make_plan({"a": after}, {"a": _held_like(before)})
    assert len(plan.rewrite) == 1 and not plan.retag


def test_an_item_the_source_no_longer_has_is_removed():
    w = _wanted_one()
    plan = make_plan({}, {"a": _held_like(w)})
    assert [h.source_ref for h in plan.remove] == ["a"]


def test_an_item_whose_text_was_emptied_is_removed_not_kept_stale():
    """ingest_connector skips an item with no text. A sync that did the same for an item
    already held would leave its old words answering questions about a page now blank."""
    held = _wanted_one(content="there used to be text")
    wanted = _wanted(FakeSource([_item(content="   ")]))
    assert wanted == {}
    assert [h.source_ref for h in make_plan(wanted, {"a": _held_like(held)}).remove] == ["a"]


def test_a_row_with_no_hash_is_treated_as_changed():
    """Written before the column existed. Guessing "unchanged" could miss an edit, which is
    the failure this feature exists to remove; the cost of guessing "changed" is one re-embed."""
    w = _wanted_one()
    plan = make_plan({"a": w}, {"a": _held_like(w, content_hash=None)})
    assert len(plan.rewrite) == 1


# --- which grants a sync may add and take away ------------------------------------------

LIVE = (True, None)
OFFICER_REVOKED = (False, None)  # revoked_by is NULL for an officer's revoke
SYNC_REVOKED = (False, SOURCE)


def test_a_new_entitlement_is_granted():
    assert grant_changes({(1, "a")}, {}) == ([(1, "a")], [])


def test_a_live_grant_the_source_still_backs_is_left_alone():
    assert grant_changes({(1, "a")}, {(1, "a"): LIVE}) == ([], [])


def test_a_live_grant_the_source_no_longer_backs_is_revoked():
    assert grant_changes(set(), {(1, "a"): LIVE}) == ([], [(1, "a")])


def test_access_a_sync_took_away_returns_when_the_source_restores_it():
    assert grant_changes({(1, "a")}, {(1, "a"): SYNC_REVOKED}) == ([(1, "a")], [])


def test_a_revoke_an_officer_made_stands():
    """The asymmetry. A sync that re-granted this ten minutes after the officer revoked it
    would undo a deliberate decision, silently, on a timer."""
    assert grant_changes({(1, "a")}, {(1, "a"): OFFICER_REVOKED}) == ([], [])


def test_a_revoked_grant_the_source_does_not_back_is_left_alone():
    assert grant_changes(set(), {(1, "a"): OFFICER_REVOKED, (2, "a"): SYNC_REVOKED}) == ([], [])


def test_everyone_holding_a_deleted_item_loses_it():
    latest = {(1, "gone"): LIVE, (2, "gone"): LIVE, (3, "gone"): OFFICER_REVOKED}
    assert grant_changes(set(), latest) == ([], [(1, "gone"), (2, "gone")])


# --- the mock sources an author can change ---------------------------------------------

@pytest.fixture
def mock_sources(tmp_path, monkeypatch):
    path = tmp_path / "mock_sources.json"
    monkeypatch.setenv("MOCK_SOURCES_PATH", str(path))
    get_settings.cache_clear()
    overlay._cache = None
    yield path
    get_settings.cache_clear()
    overlay._cache = None


def _refund():
    return next(i for i in connectors()["confluence"].list_items()
                if i.source_ref == "SUPPORT/refund-policy")


def test_with_nothing_authored_a_source_is_its_baseline(mock_sources):
    assert not mock_sources.exists()
    assert len(connectors()["confluence"].list_items()) == 3
    assert "45 days" in _refund().content


def test_an_authored_edit_is_what_the_connector_lists(mock_sources):
    edited = _refund().model_copy(update={"content": "Customers have 30 days."})
    overlay.set_item(edited)
    items = connectors()["confluence"].list_items()
    assert len(items) == 3
    assert "30 days" in _refund().content
    assert any("RESTRICTED" in i.content for i in items), "an edit leaked into other items"


def test_a_deleted_item_disappears_and_restore_brings_it_back(mock_sources):
    overlay.delete_item("confluence", "SUPPORT/refund-policy")
    assert len(connectors()["confluence"].list_items()) == 2
    assert overlay.restore("confluence", "SUPPORT/refund-policy")
    assert len(connectors()["confluence"].list_items()) == 3


def test_an_authored_permission_decides_who_may_read(mock_sources):
    connector = connectors()["confluence"]
    assert connector.check_access(ALEX, _refund())

    narrowed = ConfluencePermission(space="SUPPORT", page="refund-policy", viewer_groups=["compliance"])
    overlay.set_item(_refund(), narrowed)
    assert not connector.check_access(ALEX, _refund())
    assert connector.check_access(MARCUS, _refund())


def test_the_query_time_recheck_sees_a_change_before_any_sync(mock_sources):
    """The window the sync does not cover. Narrowing a RESTRICTED document at the source
    must stop it being answered at once; waiting for the next sync would be the mirror's
    staleness window all over again."""
    aml = next(i for i in connectors()["confluence"].list_items()
               if i.source_ref == "COMPLIANCE/aml-escalation")
    assert not source_denies(MARCUS, "confluence", "COMPLIANCE/aml-escalation")

    overlay.set_item(aml, ConfluencePermission(space="COMPLIANCE", page="aml-escalation",
                                               viewer_groups=["engineering"]))
    assert source_denies(MARCUS, "confluence", "COMPLIANCE/aml-escalation")

    overlay.delete_item("confluence", "COMPLIANCE/aml-escalation")
    assert source_denies(MARCUS, "confluence", "COMPLIANCE/aml-escalation")


def test_a_change_made_by_another_process_is_picked_up(mock_sources):
    """The CLI that authors a change and the server that syncs it are different processes,
    so the file is the only channel — read through a stat, not a parse per call."""
    assert len(connectors()["confluence"].list_items()) == 3
    mock_sources.write_text(json.dumps({"confluence:SUPPORT/refund-policy": {"op": "delete"}}))
    assert len(connectors()["confluence"].list_items()) == 2
    mock_sources.write_text(json.dumps({}))
    assert len(connectors()["confluence"].list_items()) == 3


def test_reset_puts_every_source_back(mock_sources):
    overlay.delete_item("confluence", "SUPPORT/refund-policy")
    overlay.reset()
    assert not mock_sources.exists()
    assert len(connectors()["confluence"].list_items()) == 3


def test_the_cli_edits_narrows_and_deletes(mock_sources, capsys):
    from scripts import mock_source

    ref = "confluence:SUPPORT/refund-policy"
    assert mock_source.main(["edit", ref, "45 days", "30 days"]) == 0
    assert "30 days" in _refund().content and _refund().updated_at is not None

    assert mock_source.main(["access", ref, "compliance,legal"]) == 0
    assert connectors()["confluence"].permissions_for(_refund()).viewer_groups == ["compliance", "legal"]

    assert mock_source.main(["delete", ref]) == 0
    assert mock_source.main(["list"]) == 0
    assert "(deleted)" in capsys.readouterr().out

    assert mock_source.main(["reset"]) == 0
    assert "45 days" in _refund().content


def test_the_cli_refuses_text_that_is_not_there(mock_sources):
    from scripts import mock_source

    with pytest.raises(SystemExit):
        mock_source.main(["edit", "confluence:SUPPORT/refund-policy", "no such words", "x"])
    assert not mock_sources.exists(), "a refused edit must not write anything"


# --- the officer-only trigger -----------------------------------------------------------

def _client(user, sync):
    def unused(state):
        raise AssertionError("graph must not run")

    nodes = dict.fromkeys(
        ["router", "retrieval", "sql_tool", "clarification", "synthesizer", "verifier",
         "escalation", "audit"], unused,
    )
    return TestClient(create_app(nodes=nodes, user_loader=lambda uid: user, sync=sync))


def _report(platform="confluence"):
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return SyncReport(platform=platform, trigger="manual", started_at=now, finished_at=now, updated=1)


def test_an_officer_can_run_a_sync_and_is_named_as_who_asked():
    asked = []

    def sync(trigger, user_id=None):
        asked.append((trigger, user_id))
        return [_report()]

    r = _client(MARCUS, sync).post("/admin/sync", headers=as_user(2))
    assert r.status_code == 200
    assert [x["platform"] for x in r.json()] == ["confluence"]
    assert asked == [("manual", 2)]


@pytest.mark.parametrize(("caller", "expected"), [(ALEX, 403), (None, 401)],
                         ids=["not-an-officer", "unknown-user"])
def test_nobody_else_can_start_one(caller, expected):
    def sync(trigger, user_id=None):
        raise AssertionError("a sync ran for someone who is not an officer")

    assert _client(caller, sync).post("/admin/sync", headers=as_user(1)).status_code == expected


def test_a_sync_report_carries_counts_and_never_content():
    """It is on an officer's screen, and it must stay safe to read aloud."""
    assert set(SyncReport.model_fields) == {
        "platform", "trigger", "started_at", "finished_at", "added", "updated", "removed",
        "unchanged", "grants_added", "grants_revoked", "error", "skipped",
    }
