"""Ingest connector content into the pgvector-backed document store.

This module only converts source-native permission metadata into chunk metadata.
It deliberately does not populate the live ``permissions`` table: that table
requires an authoritative source-to-user mapping which the mock connector
contract does not expose. Retrieval must continue to consult that table at
request time.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from src.connectors.base import (
    ConfluencePermission,
    DrivePermission,
    JiraPermission,
    NativePermission,
    SlackPermission,
    SourceConnector,
    SourceItem,
)

# 200 words remains comfortably below Hunyuan's 1,024-token input limit while
# leaving room for punctuation and non-English tokenisation differences.
DEFAULT_CHUNK_WORDS = 200


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class IngestionSummary:
    documents_ingested: int
    chunks_ingested: int


def chunk_text(text: str, *, max_words: int = DEFAULT_CHUNK_WORDS) -> list[str]:
    """Split text into conservative embedding-safe chunks without losing words."""
    if max_words <= 0:
        raise ValueError("max_words must be positive")

    words = text.split()
    return [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)]


def normalize_acl_tags(permission: NativePermission) -> list[str]:
    """Convert native permission shape into stable, searchable ACL metadata."""
    if isinstance(permission, ConfluencePermission):
        return permission.viewer_groups
    if isinstance(permission, JiraPermission):
        return [permission.role_required]
    if isinstance(permission, SlackPermission):
        # Public channels are available to every employee. Private-channel tags
        # preserve membership identities for the later live-permission check.
        return ["all-staff"] if not permission.is_private else [
            f"user:{member_id}" for member_id in permission.member_ids
        ]
    if isinstance(permission, DrivePermission):
        return permission.acl_entries
    raise TypeError(f"Unsupported permission type: {type(permission)!r}")


def _insert_document(connection: Any, item: SourceItem, acl_tags: list[str]) -> int:
    cursor = connection.execute(
        """
        INSERT INTO documents (title, source_platform, source_ref, dept, sensitivity, acl_tags)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (item.title, item.platform, item.source_ref, item.dept, item.sensitivity, acl_tags),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("Document insert returned no id")
    return int(row[0])


def _insert_chunk(
    connection: Any,
    *,
    document_id: int,
    content: str,
    embedding: list[float],
    acl_tags: list[str],
) -> None:
    # Passing the vector's text representation with an explicit cast works with
    # psycopg without requiring connection-global adapter registration.
    vector = "[" + ",".join(str(value) for value in embedding) + "]"
    connection.execute(
        """
        INSERT INTO document_chunks (document_id, content, embedding, acl_tags)
        VALUES (%s, %s, %s::vector, %s)
        """,
        (document_id, content, vector, acl_tags),
    )


def ingest_connector(
    connector: SourceConnector,
    connection: Any,
    embeddings: EmbeddingProvider,
    *,
    max_chunk_words: int = DEFAULT_CHUNK_WORDS,
) -> IngestionSummary:
    """Persist all connector items and their embedded, ACL-tagged chunks."""
    documents_ingested = 0
    chunks_ingested = 0

    for item in connector.list_items():
        permission = connector.permissions_for(item)
        acl_tags = normalize_acl_tags(permission)
        chunks = chunk_text(item.content, max_words=max_chunk_words)
        if not chunks:
            continue

        document_id = _insert_document(connection, item, acl_tags)
        vectors = embeddings.embed_documents(chunks)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding provider returned a vector count that does not match chunk count")

        for content, embedding in zip(chunks, vectors, strict=True):
            _insert_chunk(
                connection,
                document_id=document_id,
                content=content,
                embedding=embedding,
                acl_tags=acl_tags,
            )
            chunks_ingested += 1
        documents_ingested += 1

    connection.commit()
    return IngestionSummary(documents_ingested, chunks_ingested)
