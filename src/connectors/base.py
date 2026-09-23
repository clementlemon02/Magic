"""Source-connector contracts for secure ingestion.

Connectors expose source-native permissions. Ingestion converts those permissions
to ACL tags for search metadata; request-time authorization is always rechecked
against the live permissions table.
"""

from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from src.graph.state import Sensitivity, SourcePlatform, UserContext


class SourceItem(BaseModel):
    """Normalized content fetched from one source platform before ingestion."""

    model_config = ConfigDict(extra="forbid")

    platform: SourcePlatform
    source_ref: str
    title: str
    content: str
    dept: str
    sensitivity: Sensitivity


class ConfluencePermission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: Literal["confluence"] = "confluence"
    space: str
    page: str
    viewer_groups: list[str]


class JiraPermission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: Literal["jira"] = "jira"
    project: str
    issue: str
    role_required: str


class SlackPermission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: Literal["slack"] = "slack"
    channel: str
    is_private: bool
    member_ids: list[int]


class DrivePermission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: Literal["drive"] = "drive"
    file_id: str
    folder_id: str | None = None
    acl_entries: list[str]


NativePermission = Annotated[
    ConfluencePermission | JiraPermission | SlackPermission | DrivePermission,
    Field(discriminator="platform"),
]


@runtime_checkable
class SourceConnector(Protocol):
    """Minimal source boundary used by ingestion."""

    platform: SourcePlatform

    def list_items(self) -> list[SourceItem]: ...

    def permissions_for(self, item: SourceItem) -> NativePermission: ...

    def check_access(self, user: UserContext, item: SourceItem) -> bool: ...
