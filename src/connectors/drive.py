"""Mock Google Drive connector used by the ingestion demo."""

from src.connectors.base import DrivePermission, SourceItem
from src.graph.state import UserContext


class DriveConnector:
    platform = "drive"

    _items = [
        SourceItem(
            platform="drive",
            source_ref="file-refund-playbook",
            title="Refund Operations Playbook",
            content=(
                "Use this playbook for standard customer refund requests. First confirm the "
                "customer's identity. Then check the transaction date: chargebacks can only be "
                "contested within 45 days. Check the amount: up to SGD 2,000 you can approve it "
                "yourself, above that ask your team lead. Record the decision and the reason on "
                "the ticket, then issue the refund to the original payment method."
            ),
            dept="support",
            sensitivity="internal",
        ),
        SourceItem(
            platform="drive",
            source_ref="file-aml-evidence",
            title="AML Evidence Register",
            content=(
                "RESTRICTED - Compliance only. The evidence register holds the case files for "
                "open AML investigations, including case NGL-2291 for the account ending 7731. "
                "Each entry lists the transfers under review, the analyst assigned, and the date "
                "the Tier 2 assessment is due. Entries are retained for five years after a case "
                "closes."
            ),
            dept="compliance",
            sensitivity="restricted",
        ),
        SourceItem(
            platform="drive",
            source_ref="file-pii-standard",
            title="Customer PII Handling Standard",
            content=(
                "Customer PII covers names, phone numbers, addresses, identity document numbers "
                "and full card numbers. PII must be masked in every exported report; only the "
                "last four digits of a card number may appear. Exports that contain unmasked PII "
                "need written approval from the data protection officer and must be deleted "
                "within 30 days. On-call engineers follow the same rules during incidents."
            ),
            dept="support",
            sensitivity="internal",
        ),
    ]

    _permissions = {
        "file-refund-playbook": DrivePermission(
            file_id="file-refund-playbook", folder_id="support", acl_entries=["all-staff"]
        ),
        "file-aml-evidence": DrivePermission(
            file_id="file-aml-evidence", folder_id="compliance", acl_entries=["compliance"]
        ),
        "file-pii-standard": DrivePermission(
            file_id="file-pii-standard", folder_id="standards", acl_entries=["all-staff"]
        ),
    }

    def list_items(self) -> list[SourceItem]:
        return self._items.copy()

    def permissions_for(self, item: SourceItem) -> DrivePermission:
        return self._permissions[item.source_ref]

    def check_access(self, user: UserContext, item: SourceItem) -> bool:
        permission = self.permissions_for(item)
        return "all-staff" in permission.acl_entries or bool(
            set(user.acl_tags()) & set(permission.acl_entries)
        )
