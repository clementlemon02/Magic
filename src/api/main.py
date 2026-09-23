"""FastAPI entry point — CLAUDE.md §8 shared file, flag before editing.

Serves `POST /query` (§7). The compliance and admin routes belong to the audit
workstream and live in `src/api/compliance.py`.

The caller's identity is resolved HERE, from the database, and never taken from
the request body. A client that could send its own role or clearance_level would
walk straight past the §1 filter, since every ACL predicate downstream is built
from UserContext.
"""

import uuid
from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph.graph import build_graph, build_response
from src.graph.state import AskerResponse, GraphState, UserContext


class QueryRequest(BaseModel):
    """What a caller may send. Deliberately cannot carry role, dept or clearance."""

    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1, max_length=2000)
    user_id: int


def load_user(user_id: int) -> UserContext | None:
    """Resolve the caller from the database. Role and clearance are never client-supplied."""
    import psycopg
    from psycopg.rows import dict_row

    from src.config import get_settings

    sql = """
        SELECT u.id, r.name AS role, u.dept, r.clearance_level
        FROM users u JOIN roles r ON r.id = u.role_id
        WHERE u.id = %(user_id)s
    """
    with psycopg.connect(get_settings().database_url, row_factory=dict_row) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(sql, {"user_id": user_id})
            row = cur.fetchone()
    return UserContext(**row) if row else None


def _pending(name: str, owner: str) -> Callable[[GraphState], dict]:
    """Stand-in for a node whose workstream has not merged yet.

    Raises rather than returning empty state: a silent no-op here would look like
    a legitimately empty retrieval and produce a confident answer from nothing.
    """

    def node(state: GraphState) -> dict:
        raise NotImplementedError(f"{name} node is not implemented yet ({owner})")

    return node


def default_nodes() -> dict[str, Callable[[GraphState], dict]]:
    from src.agents.clarification import clarify_node
    from src.agents.router import route_node
    from src.agents.sql_tool import sql_tool_node
    from src.agents.synthesizer import synthesize_node
    from src.agents.verifier import verify_node

    return {
        "router": route_node,
        "clarification": clarify_node,
        "synthesizer": synthesize_node,
        "verifier": verify_node,
        "sql_tool": sql_tool_node,
        # ponytail: placeholders until these land. Swap one line each on merge.
        "retrieval": _pending("retrieval", "Chris"),
        "escalation": _pending("escalation", "Jin Hui"),
        "audit": _pending("audit", "Jin Hui"),
    }


def initial_state(query: str, user: UserContext) -> GraphState:
    return {
        "request_id": str(uuid.uuid4()),
        "query": query,
        "user": user,
        "route": "rag",
        "hop_count": 0,
        "retrieved_chunks": [],
        "permission_conflicts": [],
        "sql_result": None,
        "draft_answer": None,
        "verification": None,
        "clarification_question": None,
        "final_answer": None,
        "citations": [],
        "explanation": None,
        "escalated": False,
        "audit_events": [],
    }


def create_app(nodes: dict | None = None, user_loader=load_user) -> FastAPI:
    app = FastAPI(title="Internal Brain")
    graph = build_graph(**(nodes or default_nodes()))

    def resolve_user(request: QueryRequest) -> UserContext:
        user = user_loader(request.user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="Unknown user")
        return user

    @app.post("/query", response_model=AskerResponse)
    def query(request: QueryRequest, user: UserContext = Depends(resolve_user)) -> AskerResponse:
        state = graph.invoke(initial_state(request.query, user))
        # build_response is the only thing that shapes the reply: it cannot carry
        # `explanation`, so the §5 split holds at the boundary as well as in the graph.
        return build_response(state)

    return app


app = create_app()
