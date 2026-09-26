"""Tests for source-item ingestion without a real database or embedding API."""

import pytest

from src.connectors.base import (
    ConfluencePermission,
    DrivePermission,
    JiraPermission,
    SlackPermission,
    SourceItem,
)
from src.connectors.confluence import ConfluenceConnector
from src.ingestion.service import chunk_text, ingest_connector, normalize_acl_tags


class FakeCursor:
    def __init__(self, row=None):
        self._row = row

    def fetchone(self):
        return self._row


class FakeConnection:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.committed = False
        self._next_document_id = 1

    def execute(self, query: str, params: tuple):
        self.calls.append((query, params))
        if "INSERT INTO documents" in query:
            document_id = self._next_document_id
            self._next_document_id += 1
            return FakeCursor((document_id,))
        return FakeCursor()

    def commit(self):
        self.committed = True


class FakeEmbeddings:
    def __init__(self):
        self.requests: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(texts)
        return [[float(index), 0.5] for index, _ in enumerate(texts)]


def test_chunk_text_keeps_words_in_order():
    assert chunk_text("one two three four five", max_words=2) == ["one two", "three four", "five"]


def test_chunk_text_rejects_non_positive_chunk_sizes():
    with pytest.raises(ValueError, match="positive"):
        chunk_text("text", max_words=0)


@pytest.mark.parametrize(
    "permission, expected_tags",
    [
        (ConfluencePermission(space="SUPPORT", page="refund", viewer_groups=["support"]), ["support"]),
        (JiraPermission(project="OPS", issue="OPS-1", role_required="engineering"), ["engineering"]),
        (SlackPermission(channel="general", is_private=False, member_ids=[]), ["all-staff"]),
        (SlackPermission(channel="risk", is_private=True, member_ids=[9, 42]), ["user:9", "user:42"]),
        (DrivePermission(file_id="file-1", acl_entries=["compliance"]), ["compliance"]),
    ],
)
def test_normalize_acl_tags(permission, expected_tags):
    assert normalize_acl_tags(permission) == expected_tags


def test_ingest_connector_writes_documents_and_acl_tagged_chunks():
    connection = FakeConnection()
    embeddings = FakeEmbeddings()

    connector = ConfluenceConnector()
    summary = ingest_connector(connector, connection, embeddings, max_chunk_words=20)

    # Derived from the mock corpus, so editing its content doesn't break this test.
    items = connector.list_items()
    expected_chunks = sum(len(chunk_text(i.content, max_words=20)) for i in items)
    assert expected_chunks > len(items)  # long pages really are split
    assert summary.documents_ingested == len(items)
    assert summary.chunks_ingested == expected_chunks
    assert connection.committed
    assert len(embeddings.requests) == len(items)  # one batched call per document

    document_insert = next(params for query, params in connection.calls if "INSERT INTO documents" in query)
    assert document_insert[-1] == ["support", "all-staff"]

    chunk_inserts = [params for query, params in connection.calls if "INSERT INTO document_chunks" in query]
    assert len(chunk_inserts) == expected_chunks
    assert chunk_inserts[0][-1] == ["support", "all-staff"]


def test_ingest_rejects_embedding_count_mismatches():
    class BrokenEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return []

    connection = FakeConnection()
    with pytest.raises(RuntimeError, match="vector count"):
        ingest_connector(ConfluenceConnector(), connection, BrokenEmbeddings())
