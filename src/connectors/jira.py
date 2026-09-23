"""Mock Jira connector used by the ingestion demo."""

from src.connectors.base import JiraPermission, SourceItem
from src.graph.state import UserContext


class JiraConnector:
    platform = "jira"

    _items = [
        SourceItem(
            platform="jira",
            source_ref="SUPPORT/PLAT-101",
            title="Refund workflow incident",
            content="The refund workflow incident was resolved after queue replay.",
            dept="support",
            sensitivity="internal",
        ),
        SourceItem(
            platform="jira",
            source_ref="RISK/AML-24",
            title="AML control gap",
            content="The AML control gap requires Compliance review before closure.",
            dept="compliance",
            sensitivity="restricted",
        ),
    ]

    _permissions = {
        "SUPPORT/PLAT-101": JiraPermission(
            project="SUPPORT", issue="PLAT-101", role_required="support"
        ),
        "RISK/AML-24": JiraPermission(project="RISK", issue="AML-24", role_required="compliance"),
    }

    def list_items(self) -> list[SourceItem]:
        return self._items.copy()

    def permissions_for(self, item: SourceItem) -> JiraPermission:
        return self._permissions[item.source_ref]

    def check_access(self, user: UserContext, item: SourceItem) -> bool:
        permission = self.permissions_for(item)
        return user.role == permission.role_required or user.dept == permission.role_required
