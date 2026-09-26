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
            content=(
                "On 14 August the refund processing queue stalled after a deploy and 312 customer "
                "refunds were delayed by up to two days. The incident was resolved after the "
                "queue was replayed. Every affected customer was emailed an apology, and refunds "
                "over SGD 2,000 were re-checked by a team lead before release. Follow-up PLAT-104 "
                "adds an alert when the queue has not moved for 15 minutes."
            ),
            dept="support",
            sensitivity="internal",
        ),
        SourceItem(
            platform="jira",
            source_ref="RISK/AML-24",
            title="AML control gap: structured deposits",
            content=(
                "RESTRICTED - Compliance only. For three weeks, transaction monitoring rule TM-7 "
                "failed to flag structured deposits between SGD 9,000 and SGD 9,499 made across "
                "several days. The rule has been fixed and a lookback over the affected period "
                "found 41 accounts for manual review. The gap must not be closed until Compliance "
                "signs off the lookback, and it must not be discussed outside the Compliance "
                "team."
            ),
            dept="compliance",
            sensitivity="restricted",
        ),
        SourceItem(
            platform="jira",
            source_ref="PAY/ENG-4471",
            title="Payment outage: settlement gateway",
            content=(
                "Payment outage ENG-4471 lasted 47 minutes on 3 September and was graded SEV1. "
                "The root cause was an expired TLS certificate on the settlement gateway, which "
                "rejected every outbound settlement call. It was resolved by rotating the "
                "certificate. Follow-up tickets ENG-4472 add certificate expiry monitoring and "
                "ENG-4488 automates rotation. Support should tell affected customers that delayed "
                "payments settled automatically once the gateway recovered, and no action is "
                "needed on their side."
            ),
            dept="support",
            sensitivity="internal",
        ),
    ]

    _permissions = {
        "SUPPORT/PLAT-101": JiraPermission(
            project="SUPPORT", issue="PLAT-101", role_required="support"
        ),
        "RISK/AML-24": JiraPermission(project="RISK", issue="AML-24", role_required="compliance"),
        # Customer-facing outage, so Support reads the PAY project.
        "PAY/ENG-4471": JiraPermission(project="PAY", issue="ENG-4471", role_required="support"),
    }

    def list_items(self) -> list[SourceItem]:
        return self._items.copy()

    def permissions_for(self, item: SourceItem) -> JiraPermission:
        return self._permissions[item.source_ref]

    def check_access(self, user: UserContext, item: SourceItem) -> bool:
        permission = self.permissions_for(item)
        return user.role == permission.role_required or user.dept == permission.role_required
