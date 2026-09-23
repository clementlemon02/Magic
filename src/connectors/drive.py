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
            content="Use the refund playbook for standard customer refund requests.",
            dept="support",
            sensitivity="internal",
        ),
        SourceItem(
            platform="drive",
            source_ref="file-aml-evidence",
            title="AML Evidence Register",
            content="The evidence register contains restricted AML investigation records.",
            dept="compliance",
            sensitivity="restricted",
        ),
    ]

    _permissions = {
        "file-refund-playbook": DrivePermission(
            file_id="file-refund-playbook", folder_id="support", acl_entries=["all-staff"]
        ),
        "file-aml-evidence": DrivePermission(
            file_id="file-aml-evidence", folder_id="compliance", acl_entries=["compliance"]
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
