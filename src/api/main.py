"""FastAPI entry point — CLAUDE.md §8 shared file, flag before editing.

Serves `POST /query` (§7). The compliance and admin routes belong to the audit
workstream and live in `src/api/compliance.py`.

The caller's identity is resolved HERE, from the database, and never taken from
the request body. A client that could send its own role or clearance_level would
walk straight past the §1 filter, since every ACL predicate downstream is built
from UserContext.
"""

import logging
import uuid
from collections.abc import Callable

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.cache import PermissionAwareCache
from src.config import get_settings
from src.graph.graph import build_graph, build_response
from src.graph.state import AskerResponse, GraphState, UserContext


class QueryRequest(BaseModel):
    """What a caller may send. Deliberately cannot carry role, dept or clearance."""

    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1, max_length=2000)
    user_id: int


def load_user(user_id: int) -> UserContext | None:
    """Resolve the caller from the database. Role and clearance are never client-supplied."""
    from psycopg.rows import dict_row

    sql = """
        SELECT u.id, r.name AS role, u.dept, r.clearance_level
        FROM users u JOIN roles r ON r.id = u.role_id
        WHERE u.id = %(user_id)s
    """
    settings = get_settings()
    with psycopg.connect(
        settings.database_url,
        row_factory=dict_row,
        connect_timeout=settings.db_connect_timeout_seconds,
    ) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(sql, {"user_id": user_id})
            row = cur.fetchone()
    return UserContext(**row) if row else None


logger = logging.getLogger(__name__)

# Connection-level database faults, as opposed to a bad query.
_DB_FAULTS = (psycopg.OperationalError, psycopg.InterfaceError)


def _is_transport_fault(exc: BaseException) -> bool:
    """A network fault reaching the model, rather than a bad answer from it.

    ponytail: matched on the exception's top-level module, because the HTTP client
    behind a chat model is an implementation detail that changes with the backend and
    is not worth importing three libraries to name precisely.
    """
    root = type(exc).__module__.split(".")[0]
    return root in {"requests", "httpx", "urllib3", "http", "socket", "ssl"} or isinstance(
        exc, (TimeoutError, ConnectionError)
    )


def describe_failure(exc: BaseException) -> tuple[int, str]:
    """Map an exception to a status and a message the asker can be shown.

    None of these may be the generic refusal. A broken dependency must never be
    reported in the same words as a withheld answer: it would mislead on stage, and it
    would give that sentence a third meaning, weakening the property that a refusal
    reveals nothing (§5).
    """
    if isinstance(exc, NotImplementedError):
        return 501, "That part of the system is not built yet."
    if isinstance(exc, _DB_FAULTS):
        return 503, "The knowledge store is unavailable."
    if _is_transport_fault(exc):
        return 503, "The language model did not respond in time."
    return 503, "The service is temporarily unavailable."


def check_database() -> str:
    settings = get_settings()
    try:
        with psycopg.connect(
            settings.database_url, connect_timeout=settings.db_connect_timeout_seconds
        ):
            return "ok"
    except Exception:
        return "unavailable"


def check_llm() -> str:
    """Cheap reachability check. Only the local backend can be probed without a bill."""
    settings = get_settings()
    if settings.llm_backend == "fake":
        return "ok"
    if settings.llm_backend != "ollama":
        return "unchecked"
    try:
        import urllib.request

        with urllib.request.urlopen(
            f"{settings.ollama_base_url}/api/tags", timeout=settings.db_connect_timeout_seconds
        ) as response:
            return "ok" if response.status == 200 else "unavailable"
    except Exception:
        return "unavailable"


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
    from src.agents.retrieval import retrieval_node
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
        "retrieval": retrieval_node,
        # ponytail: placeholders until these land. Swap one line each on merge.
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


def create_app(nodes: dict | None = None, user_loader=load_user, cache=None) -> FastAPI:
    app = FastAPI(title="Internal Brain")
    graph = build_graph(**(nodes or default_nodes()))
    if cache is None and get_settings().query_cache_enabled:
        cache = PermissionAwareCache(
            similarity_threshold=get_settings().query_cache_similarity
        )

    def resolve_user(request: QueryRequest) -> UserContext:
        user = user_loader(request.user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="Unknown user")
        return user

    @app.exception_handler(Exception)
    def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Anything unplanned becomes an honest error, never a fabricated answer.

        Registered app-wide so a fault raised while resolving the caller is handled the
        same as one raised inside the graph.
        """
        status, message = describe_failure(exc)
        # exc_info=exc rather than logger.exception(): inside a FastAPI handler the
        # exception is no longer active in sys.exc_info(), so the traceback that
        # actually diagnoses a failed demo would be logged as "NoneType: None".
        logger.error("request failed: %s", type(exc).__name__, exc_info=exc)
        return JSONResponse(status_code=status, content={"detail": message})

    @app.get("/health")
    def health() -> JSONResponse:
        """Readiness for the demo. Check this before presenting, not during."""
        checks = {"database": check_database(), "llm": check_llm()}
        healthy = all(v in {"ok", "unchecked"} for v in checks.values())
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={"status": "ok" if healthy else "degraded", "checks": checks},
        )

    @app.post("/query", response_model=AskerResponse)
    def query(request: QueryRequest, user: UserContext = Depends(resolve_user)) -> AskerResponse:
        if cache is not None:
            cached = cache.get(request.query, user)
            if cached is not None:
                return cached

        state = graph.invoke(initial_state(request.query, user))
        # build_response is the only thing that shapes the reply: it cannot carry
        # `explanation`, so the §5 split holds at the boundary as well as in the graph.
        response = build_response(state)

        # Refusals are never cached. They are the security-critical path, every one
        # of them owes an `escalations` row (§4), and serving one from memory would
        # skip the graph that writes it. Answers are cached; a hit still owes an
        # audit row, which the audit workstream has to emit here — see the PR.
        if cache is not None and not state.get("escalated"):
            cache.put(request.query, user, response)
        return response

    return app


app = create_app()
