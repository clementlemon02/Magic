"""FastAPI entry point — CLAUDE.md §8 shared file, flag before editing.

Serves `POST /query` (§7). The compliance and admin routes live in
`src/api/compliance.py` and are mounted here.

The caller's identity is resolved HERE, from the database, and never taken from
the request body. A client that could send its own role or clearance_level would
walk straight past the §1 filter, since every ACL predicate downstream is built
from UserContext.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.auth import build_dependencies, issue_token, verify_password
from src.api.compliance import build_router
from src.api.ui import build_router as build_ui_router
from src.cache import PermissionAwareCache
from src.config import get_settings
from src.graph.graph import build_graph, build_response
from src.graph.state import AskerResponse, AuditEvent, GraphState, Stage, UserContext


class QueryRequest(BaseModel):
    """What a caller may send: the question, and nothing about themselves.

    `user_id` used to be a field here, unchecked against anything, so any client
    could ask as anyone — and identity is the input every ACL predicate downstream
    is built from. The caller comes from a signed token now (src/api/auth.py).
    """

    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1, max_length=2000)


class LoginRequest(BaseModel):
    model_config = {"extra": "forbid"}

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class Me(BaseModel):
    """Who the token says you are, answered by the database."""

    id: int
    role: str
    dept: str
    clearance_level: int


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
    # Role and clearance come from HERE on every request and never from the token —
    # see src/api/auth.py. That is what stops a bearer token asserting a role.
    return UserContext(**row) if row else None


def load_credentials(email: str) -> tuple[int, str | None] | None:
    """The id and stored hash for an email, for sign-in only.

    Separate from `load_user` so nothing on the request path can reach a hash, and
    so `UserContext` never carries one.
    """
    settings = get_settings()
    with psycopg.connect(
        settings.database_url, connect_timeout=settings.db_connect_timeout_seconds
    ) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, password_hash FROM users WHERE lower(email) = lower(%(email)s)",
                {"email": email},
            )
            row = cur.fetchone()
    return (row[0], row[1]) if row else None


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


async def _warm(settings) -> None:
    """Load the model before the first question arrives, not during it.

    Fire and forget: the server is serving as soon as it binds, and a failure here
    is a slow first answer, never a broken one — so every error is swallowed apart
    from a log line. /health does not wait on it either, since a warm model is a
    nicety and a reachable API is not.
    """
    if not settings.warm_on_startup:
        return

    def once() -> None:
        from src.llm.factory import get_chat_model, get_embeddings

        get_embeddings().embed_query("warm")
        get_chat_model().invoke("Reply with the single word: ready")

    try:
        await run_in_threadpool(once)
        logger.info("model warmed")
    except Exception as error:  # noqa: BLE001 — a cold model is not a failed start
        logger.warning("warm-up skipped: %s", error)


def create_app(
    nodes: dict | None = None,
    user_loader=load_user,
    credential_loader=load_credentials,
    cache=None,
    *,
    refusal_deadline: float | None = None,
    clock=time.monotonic,
    sleeper=asyncio.sleep,
) -> FastAPI:
    """Build the app. `clock` and `sleeper` are injectable so padding is testable
    without sleeping; `refusal_deadline` overrides the configured value, 0 disables it.
    """
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Not awaited: the server should be answering before the model finishes
        # loading, and a warm-up that fails is a slow first answer, not a bad start.
        task = asyncio.create_task(_warm(get_settings()))
        yield
        task.cancel()

    app = FastAPI(title="Internal Brain", lifespan=lifespan)
    nodes = nodes or default_nodes()
    graph = build_graph(**nodes)
    # Built before the routers: both the compliance router and /query depend on it.
    caller, officer = build_dependencies(user_loader)
    app.include_router(build_router(user_loader, officer=officer))
    app.include_router(build_ui_router())
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

    @app.post("/auth/login", response_model=Token)
    async def login(body: LoginRequest) -> Token:
        settings = get_settings()

        def check() -> int | None:
            found = credential_loader(body.email)
            # verify_password is run even when the email is unknown, against a hash
            # that cannot match, so a wrong address and a wrong password cost the
            # same time. Otherwise the response time enumerates accounts.
            user_id, stored = found if found else (None, None)
            return user_id if verify_password(body.password, stored) else None

        user_id = await run_in_threadpool(check)
        if user_id is None:
            # One message for both, so it never says which half was wrong.
            raise HTTPException(status_code=401, detail="That email and password do not match.")
        return Token(
            access_token=issue_token(user_id),
            expires_in=settings.auth_token_ttl_minutes * 60,
        )

    @app.get("/auth/me", response_model=Me)
    async def me(user: UserContext = Depends(caller)) -> Me:
        return Me(**user.model_dump())

    @app.middleware("http")
    async def stamp_arrival(request: Request, call_next):
        """Open the constant-time envelope at the very start of the request.

        The handler used to call clock() itself, which was early enough when the
        caller was resolved inside it. A dependency runs BEFORE the handler, so
        authenticating there moved a variable, database-backed step outside the
        padding — and a step whose duration varies is exactly what the padding
        exists to hide (design doc §6.3). Middleware runs before dependencies.
        """
        request.state.arrived = clock()
        return await call_next(request)

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
    async def query(
        request: QueryRequest, raw: Request, user: UserContext = Depends(caller)
    ) -> AskerResponse:
        # Stamped by stamp_arrival before any dependency ran, so authentication and
        # the cache lookup are both inside the envelope.
        started = getattr(raw.state, "arrived", None)
        if started is None:  # pragma: no cover - middleware always runs in the app
            started = clock()

        # Async handler, blocking work on the threadpool: padding then awaits on the
        # event loop and holds no worker. A sync sleep would hold one of ~40 for the
        # full deadline, and parallel restricted questions would exhaust the pool.
        # `user` is already resolved: the dependency verified the token and read the
        # row before this handler ran.

        if cache is not None:
            entry = await run_in_threadpool(cache.lookup, request.query, user)
            if entry is not None:
                cached = entry.response
                # Served without running the graph, so audited here: an answer that
                # left no trail would be the one gap in "every answer is on record".
                hit = {
                    **initial_state(request.query, user),
                    # The route that produced the entry, not initial_state's `rag`
                    # placeholder — the Router did not run, so the placeholder would
                    # log a cached sql or decline request as a rag one. An older entry
                    # stored before routes were recorded keeps the placeholder.
                    **({"route": entry.route} if entry.route else {}),
                    "final_answer": cached.text,
                    "citations": cached.citations,
                    "audit_events": [AuditEvent(
                        event_type="cache_hit", payload={}, occurred_at=datetime.now(UTC)
                    )],
                }
                await run_in_threadpool(nodes["audit"], hit)
                # Replace the trace, never pass the cached one through: it describes
                # the request that filled the cache, not this one, and the UI would
                # show seconds of retrieval and synthesis that did not happen. One
                # stage for what actually ran.
                return cached.model_copy(update={
                    "trace": [Stage(node="cache", hop=0, ms=(clock() - started) * 1000)]
                })

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
            await run_in_threadpool(
                cache.put, request.query, user, response, state.get("route")
            )
        return response

    return app


app = create_app()
