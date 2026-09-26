"""FastAPI entry point — CLAUDE.md §8 shared file, flag before editing.

Serves `POST /query` (§7). The compliance and admin routes live in
`src/api/compliance.py` and are mounted here.

The caller's identity is resolved HERE, from the database, and never taken from
the request body. A client that could send its own role or clearance_level would
walk straight past the §1 filter, since every ACL predicate downstream is built
from UserContext.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.compliance import build_router
from src.cache import PermissionAwareCache
from src.config import get_settings
from src.graph.graph import build_graph, build_response
from src.graph.state import AskerResponse, AuditEvent, GraphState, UserContext


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
    would give that sentence a third meaning, weakening the property that a refusal's
    content reveals nothing (§5). Its timing is handled separately, by hold_refusal.
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


def default_nodes() -> dict[str, Callable[[GraphState], dict]]:
    from src.agents.audit import audit_node
    from src.agents.clarification import clarify_node
    from src.agents.escalation import escalation_node
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
        "escalation": escalation_node,
        "audit": audit_node,
    }


def initial_state(query: str, user: UserContext) -> GraphState:
    return {
        "request_id": str(uuid.uuid4()),
        "query": query,
        "user": user,
        "route": "rag",
        "hop_count": 0,
        "started_at": time.monotonic(),
        "retrieved_chunks": [],
        "evidence_exhausted": False,
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


def create_app(
    nodes: dict | None = None,
    user_loader=load_user,
    cache=None,
    *,
    refusal_deadline: float | None = None,
    clock=time.monotonic,
    sleeper=asyncio.sleep,
) -> FastAPI:
    """Build the app. `clock` and `sleeper` are injectable so padding is testable
    without sleeping; `refusal_deadline` overrides the configured value, 0 disables it.
    """
    app = FastAPI(title="Internal Brain")
    nodes = nodes or default_nodes()
    graph = build_graph(**nodes)
    app.include_router(build_router(user_loader))
    settings = get_settings()
    if cache is None and settings.query_cache_enabled:
        cache = PermissionAwareCache(similarity_threshold=settings.query_cache_similarity)
    if refusal_deadline is None:
        refusal_deadline = (
            settings.refusal_deadline_seconds if settings.refusal_padding_enabled else 0.0
        )

    async def hold_refusal(started: float) -> None:
        """Keep a refusal until the deadline, so its timing can't say why it happened.

        A permission conflict short-circuits the graph and a refusal for lack of
        evidence runs up to three hops, so unpadded they were separable from a single
        timing measurement (docs/design/constant-time-refusal.md §4). A refusal that is
        already past the deadline is not delayed further — it leaks in one direction
        only, and evals/refusal_timing.py reports how often that happens.
        """
        remaining = refusal_deadline - (clock() - started)
        if remaining > 0:
            await sleeper(remaining)

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
    async def query(request: QueryRequest) -> AskerResponse:
        # The clock starts before the caller is resolved and before the cache lookup:
        # both vary and both precede the graph, so starting later would leave them
        # outside the padded envelope (design doc §6.3).
        started = clock()

        # Async handler, blocking work on the threadpool: padding then awaits on the
        # event loop and holds no worker. A sync sleep would hold one of ~40 for the
        # full deadline, and parallel restricted questions would exhaust the pool.
        user = await run_in_threadpool(resolve_user, request)

        if cache is not None:
            cached = await run_in_threadpool(cache.get, request.query, user)
            if cached is not None:
                # Served without running the graph, so audited here: an answer that
                # left no trail would be the one gap in "every answer is on record".
                hit = {
                    **initial_state(request.query, user),
                    "final_answer": cached.text,
                    "citations": cached.citations,
                    "audit_events": [AuditEvent(
                        event_type="cache_hit", payload={}, occurred_at=datetime.now(UTC)
                    )],
                }
                await run_in_threadpool(nodes["audit"], hit)
                return cached

        state = await run_in_threadpool(graph.invoke, initial_state(request.query, user))
        # build_response is the only thing that shapes the reply: it cannot carry
        # `explanation`, so the §5 split holds at the boundary as well as in the graph.
        response = build_response(state)

        if state.get("escalated"):
            # Refusals are padded and never cached: they are the security-critical
            # path, and every one owes an `escalations` row (§4) that serving from
            # memory would skip.
            await hold_refusal(started)
            return response

        # Answers are cached, and not padded — answer versus refusal is already
        # visible in the text.
        if cache is not None:
            await run_in_threadpool(cache.put, request.query, user, response)
        return response

    return app


app = create_app()
