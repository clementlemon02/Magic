"""Local Postgres integration checks for ACL-safe pgvector retrieval.

Run explicitly after starting the local Docker database:
RUN_DATABASE_INTEGRATION=1 python -m pytest tests/test_retrieval_integration.py -q
"""

import os

import psycopg
import pytest
from psycopg.rows import dict_row

from src.agents.retrieval import retrieve
from src.config import get_settings
from src.graph.state import UserContext

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DATABASE_INTEGRATION") != "1",
    reason="set RUN_DATABASE_INTEGRATION=1 after starting the local Docker database",
)


class FixedEmbeddings:
    """Avoids Hunyuan credentials while exercising real pgvector SQL."""

    def embed_query(self, text: str) -> list[float]:
        return [1.0, *([0.0] * 1023)]


def _vector(first: float, second: float = 0.0) -> str:
    values = [first, second, *([0.0] * 1022)]
    return "[" + ",".join(str(value) for value in values) + "]"


@pytest.fixture
def connection():
    """Yield an uncommitted connection so all fixture data is rolled back."""
    try:
        conn = psycopg.connect(get_settings().database_url)
    except psycopg.OperationalError as error:
        pytest.skip(f"local database is unavailable: {error}")

    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


def _insert_user(connection, *, name: str, role: str, dept: str, clearance_level: int) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO roles (name, clearance_level) VALUES (%s, %s) RETURNING id",
            (role, clearance_level),
        )
        role_id = cursor.fetchone()[0]
        cursor.execute(
            """
            INSERT INTO users (name, email, role_id, dept)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (name, f"{name}@integration.test", role_id, dept),
        )
        return int(cursor.fetchone()[0])


def _insert_document(connection, *, title: str, source_ref: str, sensitivity: str, acl_tags: list[str], vector: str) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (title, source_platform, source_ref, dept, sensitivity, acl_tags)
            VALUES (%s, 'confluence', %s, %s, %s, %s)
            RETURNING id
            """,
            (title, source_ref, "compliance" if sensitivity == "restricted" else "support", sensitivity, acl_tags),
        )
        document_id = int(cursor.fetchone()[0])
        cursor.execute(
            """
            INSERT INTO document_chunks (document_id, content, embedding, acl_tags)
            VALUES (%s, %s, %s::vector, %s)
            """,
            (document_id, f"Evidence from {title}.", vector, acl_tags),
        )
        return document_id


def _grant(connection, *, user_id: int, source_ref: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO permissions (user_id, source_platform, source_ref)
            VALUES (%s, 'confluence', %s)
            """,
            (user_id, source_ref),
        )


@pytest.fixture
def seeded_access(connection):
    restricted_user_id = _insert_user(
        connection,
        name="restricted-support-user",
        role="integration-support-role",
        dept="support",
        clearance_level=0,
    )
    elevated_user_id = _insert_user(
        connection,
        name="elevated-compliance-user",
        role="integration-compliance-role",
        dept="compliance",
        clearance_level=1,
    )

    _insert_document(
        connection,
        title="Integration Refund Policy",
        source_ref="INTEGRATION/refund-policy",
        sensitivity="internal",
        acl_tags=["support", "compliance"],
        vector=_vector(0.8, 0.6),
    )
    _insert_document(
        connection,
        title="Integration AML Procedure",
        source_ref="INTEGRATION/aml-procedure",
        sensitivity="restricted",
        acl_tags=["compliance"],
        vector=_vector(1.0),
    )

    _grant(connection, user_id=restricted_user_id, source_ref="INTEGRATION/refund-policy")
    _grant(connection, user_id=elevated_user_id, source_ref="INTEGRATION/refund-policy")
    _grant(connection, user_id=elevated_user_id, source_ref="INTEGRATION/aml-procedure")

    return {
        "restricted": UserContext(
            id=restricted_user_id,
            role="integration-support-role",
            dept="support",
            clearance_level=0,
        ),
        "elevated": UserContext(
            id=elevated_user_id,
            role="integration-compliance-role",
            dept="compliance",
            clearance_level=1,
        ),
    }


def _execute(connection):
    def execute(sql, params):
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()

    return execute


def _retrieve_for(user, connection):
    return retrieve(
        "What is the procedure?",
        user,
        _execute(connection),
        FixedEmbeddings(),
        top_k=6,
        min_score=0.72,
        conflict_score_margin=0.05,
    )


def test_restricted_user_receives_permitted_content_and_a_metadata_only_conflict(
    connection, seeded_access
):
    chunks, conflicts = _retrieve_for(seeded_access["restricted"], connection)

    assert [chunk.citation.source_ref for chunk in chunks] == ["INTEGRATION/refund-policy"]
    assert [conflict.source_ref for conflict in conflicts] == ["INTEGRATION/aml-procedure"]
    assert "content" not in conflicts[0].model_fields


def test_elevated_user_receives_both_authorized_documents_without_a_conflict(connection, seeded_access):
    chunks, conflicts = _retrieve_for(seeded_access["elevated"], connection)

    assert {chunk.citation.source_ref for chunk in chunks} == {
        "INTEGRATION/refund-policy",
        "INTEGRATION/aml-procedure",
    }
    assert conflicts == []
