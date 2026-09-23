"""Tests for source-native connector contracts."""

import pytest
from pydantic import TypeAdapter, ValidationError

from src.connectors.base import (
    ConfluencePermission,
    NativePermission,
    SourceConnector,
    SourceItem,
)
from src.connectors.confluence import ConfluenceConnector
from src.connectors.drive import DriveConnector
from src.connectors.jira import JiraConnector
from src.connectors.slack import SlackConnector
from src.graph.state import UserContext


def _item() -> SourceItem:
    return SourceItem(
        platform="confluence",
        source_ref="SUPPORT/refund-policy",
        title="Refund Policy",
        content="Refund requests must be reviewed within five business days.",
        dept="support",
        sensitivity="internal",
    )


def test_source_item_rejects_precomputed_acl_tags():
    """ACL tags belong to ingestion, not to an untrusted source item."""
    with pytest.raises(ValidationError):
        SourceItem(
            platform="confluence",
            source_ref="SUPPORT/refund-policy",
            title="Refund Policy",
            content="Internal policy text.",
            dept="support",
            sensitivity="internal",
            acl_tags=["support"],
        )


def test_native_permission_is_discriminated_by_platform():
    permission = TypeAdapter(NativePermission).validate_python(
        {
            "platform": "confluence",
            "space": "SUPPORT",
            "page": "refund-policy",
            "viewer_groups": ["support", "all-staff"],
        }
    )

    assert isinstance(permission, ConfluencePermission)
    assert permission.viewer_groups == ["support", "all-staff"]


def test_connector_contract_is_runtime_checkable():
    class FakeConnector:
        platform = "confluence"

        def list_items(self) -> list[SourceItem]:
            return [_item()]

        def permissions_for(self, item: SourceItem) -> ConfluencePermission:
            return ConfluencePermission(
                space="SUPPORT",
                page="refund-policy",
                viewer_groups=["support"],
            )

        def check_access(self, user: UserContext, item: SourceItem) -> bool:
            return user.dept == "support"

    connector = FakeConnector()
    assert isinstance(connector, SourceConnector)
    assert connector.check_access(
        UserContext(id=1, role="agent", dept="support", clearance_level=0),
        _item(),
    )


@pytest.mark.parametrize(
    "connector_class, expected_platform",
    [
        (ConfluenceConnector, "confluence"),
        (JiraConnector, "jira"),
        (SlackConnector, "slack"),
        (DriveConnector, "drive"),
    ],
)
def test_mock_connectors_supply_platform_specific_items(connector_class, expected_platform):
    connector = connector_class()

    assert isinstance(connector, SourceConnector)
    assert connector.platform == expected_platform
    assert all(item.platform == expected_platform for item in connector.list_items())


def test_confluence_restricts_the_aml_page_to_compliance():
    connector = ConfluenceConnector()
    restricted_item = connector.list_items()[1]

    assert not connector.check_access(
        UserContext(id=1, role="support", dept="support", clearance_level=0), restricted_item
    )
    assert connector.check_access(
        UserContext(id=9, role="compliance", dept="compliance", clearance_level=1), restricted_item
    )


def test_jira_uses_the_native_required_role():
    connector = JiraConnector()
    restricted_item = connector.list_items()[1]

    assert not connector.check_access(
        UserContext(id=1, role="support", dept="support", clearance_level=0), restricted_item
    )
    assert connector.check_access(
        UserContext(id=9, role="compliance", dept="compliance", clearance_level=1), restricted_item
    )


def test_slack_private_channel_requires_membership():
    connector = SlackConnector()
    restricted_item = connector.list_items()[1]

    assert not connector.check_access(
        UserContext(id=1, role="compliance", dept="compliance", clearance_level=1), restricted_item
    )
    assert connector.check_access(
        UserContext(id=9, role="compliance", dept="compliance", clearance_level=1), restricted_item
    )


def test_drive_restricts_the_evidence_register_to_compliance():
    connector = DriveConnector()
    restricted_item = connector.list_items()[1]

    assert not connector.check_access(
        UserContext(id=1, role="support", dept="support", clearance_level=0), restricted_item
    )
    assert connector.check_access(
        UserContext(id=9, role="compliance", dept="compliance", clearance_level=1), restricted_item
    )
