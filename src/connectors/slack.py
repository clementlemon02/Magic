"""Mock Slack connector used by the ingestion demo."""

from src.connectors.base import SlackPermission, SourceItem
from src.graph.state import UserContext


class SlackConnector:
    platform = "slack"

    _items = [
        SourceItem(
            platform="slack",
            source_ref="support-updates/1700000000.000001",
            title="#support-updates: refund backlog",
            content="The refund backlog is cleared; normal response times have resumed.",
            dept="support",
            sensitivity="internal",
        ),
        SourceItem(
            platform="slack",
            source_ref="compliance-alerts/1700000000.000002",
            title="#compliance-alerts: AML review",
            content="A restricted AML review is in progress.",
            dept="compliance",
            sensitivity="restricted",
        ),
    ]

    _permissions = {
        "support-updates/1700000000.000001": SlackPermission(
            channel="support-updates", is_private=False, member_ids=[]
        ),
        "compliance-alerts/1700000000.000002": SlackPermission(
            channel="compliance-alerts", is_private=True, member_ids=[9, 42]
        ),
    }

    def list_items(self) -> list[SourceItem]:
        return self._items.copy()

    def permissions_for(self, item: SourceItem) -> SlackPermission:
        return self._permissions[item.source_ref]

    def check_access(self, user: UserContext, item: SourceItem) -> bool:
        permission = self.permissions_for(item)
        return not permission.is_private or user.id in permission.member_ids
