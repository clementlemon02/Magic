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
            content=(
                "Customers have 45 days from the transaction date to contest a chargeback. "
                "Disputes raised after 45 days are declined unless a manager approves an "
                "override, which must be recorded on the ticket with the reason. Support must "
                "review every refund request within five business days of receipt. Refunds up to "
                "SGD 2,000 can be approved by the handling agent; refunds above SGD 2,000 need "
                "team lead approval before they are issued. Refunds are always returned to the "
                "original payment method, never to a different account, even if the customer "
                "asks. If a refund request mentions fraud or an account the customer does not "
                "recognise, stop and route it to the fraud queue instead of processing it."
            ),
            dept="support",
            sensitivity="internal",
        ),
        SourceItem(
            platform="confluence",
            source_ref="COMPLIANCE/aml-escalation",
            title="AML Escalation Procedure",
            content=(
                "RESTRICTED - Compliance only. Any single transfer of SGD 9,500 or more to a "
                "counterparty in a high-risk jurisdiction triggers a Tier 2 review under Project "
                "Nightingale. The reviewing analyst has two business days to complete the Tier 2 "
                "assessment. If the assessment finds reasonable grounds for suspicion, a "
                "Suspicious Transaction Report is filed within 15 business days. Analysts must "
                "not tell the customer that a review is underway, and must not mention the review "
                "in any support ticket, because doing so is tipping off. Accounts under Tier 2 "
                "review are placed on enhanced monitoring for 90 days."
            ),
            dept="compliance",
            sensitivity="restricted",
        ),
        SourceItem(
            platform="confluence",
            source_ref="ENG/incident-response",
            title="Incident Response Runbook",
            content=(
                "Incidents are graded SEV1 to SEV3. A SEV1 is any outage that stops customers "
                "from paying or receiving money; it pages the on-call engineer and the incident "
                "commander immediately, and a status update goes to #support-updates every 30 "
                "minutes until it is resolved. SEV2 covers degraded service with a workaround and "
                "pages during business hours only. Every SEV1 and SEV2 needs a written postmortem "
                "within five business days, with follow-up tickets filed in Jira and linked from "
                "the postmortem."
            ),
            dept="engineering",
            sensitivity="internal",
        ),
    ]

    _permissions = {
        "SUPPORT/refund-policy": ConfluencePermission(
            space="SUPPORT", page="refund-policy", viewer_groups=["support", "all-staff"]
        ),
        "COMPLIANCE/aml-escalation": ConfluencePermission(
            space="COMPLIANCE", page="aml-escalation", viewer_groups=["compliance"]
        ),
        "ENG/incident-response": ConfluencePermission(
            space="ENG", page="incident-response", viewer_groups=["all-staff"]
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
