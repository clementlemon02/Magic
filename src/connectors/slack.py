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
            content=(
                "Update from the support leads: the refund backlog from the PLAT-101 queue stall "
                "is fully cleared, and normal response times have resumed. Refund requests are "
                "back to being reviewed within five business days."
            ),
            dept="support",
            sensitivity="internal",
        ),
        SourceItem(
            platform="slack",
            source_ref="compliance-alerts/1700000000.000002",
            title="#compliance-alerts: Tier 2 review",
            content=(
                "Heads up: account ending 7731 has entered Tier 2 review after two transfers to a "
                "high-risk jurisdiction. Enhanced monitoring is on. Do not contact the customer, "
                "and keep this out of support tickets."
            ),
            dept="compliance",
            sensitivity="restricted",
        ),
        SourceItem(
            platform="slack",
            source_ref="support-eng/1726000000.000100",
            title="#support-eng: on-call PII access",
            content=(
                "Reminder for on-call: if you need to see customer PII while handling an "
                "incident, request it through the standing approval in #support-eng and wait for "
                "a lead to approve. Access expires after 8 hours. Never paste customer PII into "
                "Jira tickets or Slack threads; link to the record in the admin console instead."
            ),
            dept="support",
            sensitivity="internal",
        ),
    ]

    _permissions = {
        "support-updates/1700000000.000001": SlackPermission(
            channel="support-updates", is_private=False, member_ids=[]
        ),
        # Marcus (2) is a member, so the compliance persona can read it in the demo.
        "compliance-alerts/1700000000.000002": SlackPermission(
            channel="compliance-alerts", is_private=True, member_ids=[2, 9, 42]
        ),
        "support-eng/1726000000.000100": SlackPermission(
            channel="support-eng", is_private=False, member_ids=[]
        ),
    }

    def list_items(self) -> list[SourceItem]:
        return self._items.copy()

    def permissions_for(self, item: SourceItem) -> SlackPermission:
        return self._permissions[item.source_ref]

    def check_access(self, user: UserContext, item: SourceItem) -> bool:
        permission = self.permissions_for(item)
        return not permission.is_private or user.id in permission.member_ids
