"""ACL-safe pgvector retrieval.

Content is selected only by ``FILTERED_SEARCH``. The separate conflict search
never selects chunk text, so a restricted match cannot reach graph state that
feeds the Synthesizer.
"""

from typing import Any, Protocol

from src.graph.state import Chunk, Citation, GraphState, PermConflict, UserContext

FILTERED_SEARCH = """
    SELECT
        chunk.id,
        document.id AS document_id,
        chunk.content,
        chunk.acl_tags,
        document.title,
        document.source_platform,
        document.source_ref,
        1 - (chunk.embedding <=> %(embedding)s::vector) AS score
    FROM document_chunks AS chunk
    JOIN documents AS document ON document.id = chunk.document_id
    JOIN permissions AS permission
      ON permission.source_platform = document.source_platform
     AND permission.source_ref = document.source_ref
     AND permission.user_id = %(user_id)s
     AND permission.revoked_at IS NULL
    WHERE chunk.acl_tags && %(acl_tags)s
      AND 1 - (chunk.embedding <=> %(embedding)s::vector) >= %(min_score)s
    ORDER BY chunk.embedding <=> %(embedding)s::vector
    LIMIT %(top_k)s
"""

# Deliberately excludes chunk.content. This is a metadata-only path used solely
# to decide whether the graph must escalate before synthesis.
RESTRICTED_CONFLICT_SEARCH = """
    SELECT
        document.id AS document_id,
        document.source_platform,
        document.source_ref,
        document.sensitivity,
        1 - (chunk.embedding <=> %(embedding)s::vector) AS score
    FROM document_chunks AS chunk
    JOIN documents AS document ON document.id = chunk.document_id
    WHERE document.sensitivity = 'restricted'
      AND NOT EXISTS (
          SELECT 1
          FROM permissions AS permission
          WHERE permission.source_platform = document.source_platform
            AND permission.source_ref = document.source_ref
            AND permission.user_id = %(user_id)s
            AND permission.revoked_at IS NULL
      )
      AND 1 - (chunk.embedding <=> %(embedding)s::vector) >= %(min_score)s
    ORDER BY chunk.embedding <=> %(embedding)s::vector
    LIMIT %(top_k)s
"""


class EmbeddingProvider(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"


def _caller_acl_tags(user: UserContext) -> list[str]:
    """Include private-source membership tags alongside role and department."""
    return [*user.acl_tags(), f"user:{user.id}"]


def _as_chunk(row: dict[str, Any]) -> Chunk:
    document_id = int(row["document_id"])
    return Chunk(
        id=int(row["id"]),
        document_id=document_id,
        content=str(row["content"]),
        acl_tags=list(row["acl_tags"]),
        score=float(row["score"]),
        citation=Citation(
            document_id=document_id,
            title=str(row["title"]),
            source_platform=row["source_platform"],
            source_ref=str(row["source_ref"]),
        ),
    )


def retrieve(
    query: str,
    user: UserContext,
    execute,
    embeddings: EmbeddingProvider,
    *,
    top_k: int,
    min_score: float,
    conflict_score_margin: float,
) -> tuple[list[Chunk], list[PermConflict]]:
    """Return permitted chunks and metadata-only higher-ranked conflicts."""
    vector = _vector_literal(embeddings.embed_query(query))
    params = {
        "embedding": vector,
        "user_id": user.id,
        "acl_tags": _caller_acl_tags(user),
        "top_k": top_k,
        "min_score": min_score,
    }

    permitted = [_as_chunk(row) for row in execute(FILTERED_SEARCH, params)]
    restricted = execute(RESTRICTED_CONFLICT_SEARCH, params)
    best_permitted_score = permitted[0].score if permitted else None

    conflicts: list[PermConflict] = []
    for row in restricted:
        score = float(row["score"])
        # With no permitted evidence, any relevant inaccessible restricted result
        # is a conflict. Otherwise it must strictly exceed the permitted result by
        # the configured safety margin.
        is_higher_ranked = best_permitted_score is None or (
            score > best_permitted_score + conflict_score_margin + 1e-9
        )
        if is_higher_ranked:
            conflicts.append(
                PermConflict(
                    document_id=int(row["document_id"]),
                    source_platform=row["source_platform"],
                    source_ref=str(row["source_ref"]),
                    sensitivity=row["sensitivity"],
                    score_margin=score - (best_permitted_score or 0.0),
                )
            )
    return permitted, conflicts


def _psycopg_execute(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    import psycopg
    from psycopg.rows import dict_row

    from src.config import get_settings

    with psycopg.connect(get_settings().database_url, row_factory=dict_row) as connection:
        connection.read_only = True
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()


def _query_for_hop(state: GraphState) -> str:
    """Use verifier feedback as a safe reformulation hint on subsequent hops."""
    verification = state.get("verification")
    if not verification or not verification.unsupported:
        return state["query"]
    hints = "; ".join(verification.unsupported)
    return f"{state['query']}\nFind evidence for: {hints}"


def retrieval_node(state: GraphState, embeddings=None, execute=None) -> dict:
    """LangGraph node: retrieve one ACL-safe hop and increment its counter."""
    if embeddings is None:
        from src.llm.factory import get_embeddings

        embeddings = get_embeddings()
    if execute is None:
        execute = _psycopg_execute

    from src.config import get_settings

    settings = get_settings()
    chunks, conflicts = retrieve(
        _query_for_hop(state),
        state["user"],
        execute,
        embeddings,
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
        conflict_score_margin=settings.permission_conflict_score_margin,
    )
    return {
        "retrieved_chunks": chunks,
        "permission_conflicts": conflicts,
        "hop_count": state.get("hop_count", 0) + 1,
    }
