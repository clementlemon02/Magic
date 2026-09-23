"""Mock Confluence connector used by the ingestion demo."""

from src.connectors.base import ConfluencePermission, SourceItem
from src.graph.state import UserContext


class ConfluenceConnector:
    platform = "confluence"

    _items = [
        SourceItem(
            platform="confluence",
            source_ref="SUPPORT/refund-policy",
            title="Refund and Chargeback Policy",
            content="Support must review refund requests within five business days.",
            dept="support",
            sensitivity="internal",
        ),
        SourceItem(
            platform="confluence",
            source_ref="COMPLIANCE/aml-escalation",
            title="AML Escalation Procedure",
            content="Escalate transactions that meet the AML review threshold.",
            dept="compliance",
            sensitivity="restricted",
        ),
    ]

    _permissions = {
        "SUPPORT/refund-policy": ConfluencePermission(
            space="SUPPORT", page="refund-policy", viewer_groups=["support", "all-staff"]
        ),
        "COMPLIANCE/aml-escalation": ConfluencePermission(
            space="COMPLIANCE", page="aml-escalation", viewer_groups=["compliance"]
        ),
    }

    def list_items(self) -> list[SourceItem]:
        return self._items.copy()

    def permissions_for(self, item: SourceItem) -> ConfluencePermission:
        return self._permissions[item.source_ref]

    def check_access(self, user: UserContext, item: SourceItem) -> bool:
        permission = self.permissions_for(item)
        return "all-staff" in permission.viewer_groups or bool(
            set(user.acl_tags()) & set(permission.viewer_groups)
        )
