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

from src.api.auth import SlidingWindow, build_dependencies, issue_token, verify_password
from src.api.compliance import build_router
from src.api.ui import build_router as build_ui_router
from src.cache import PermissionAwareCache
from src.config import get_settings
from src.connectors import is_restricted
from src.graph.graph import build_graph, build_response
from src.graph.state import AskerResponse, AuditEvent, GraphState, Stage, UserContext
from src.ingestion.sync import last_synced, sync_sources
from src.llm.factory import ModelUnavailable


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


# _is_transport_fault used to live here and walk the exception's cause chain. It
# could not work from this end: anyio carries a threadpool fault through a task group
# and overwrites __context__, so what arrives is `ValueError -> ExceptionGroup ->
# ValueError -> ...` with the requests error destroyed. The classification moved to
# src/llm/factory.py, where the chain is still intact, and arrives here as a TYPE —
# which survives any number of re-raises.


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
    # ModelUnavailable is the factory's own verdict, made where the cause chain was
    # still intact. TimeoutError is kept alongside it because a bare one can only be
    # the model by this point — database faults are matched above, and nothing else
    # on the request path waits on a socket.
    if isinstance(exc, (ModelUnavailable, TimeoutError)):
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


def _rests_on_restricted(response: AskerResponse) -> bool:
    """Does this answer cite a document the source holds as restricted? See is_restricted."""
    return any(is_restricted(c.source_platform, c.source_ref) for c in response.citations)


async def _migrate(settings) -> None:
    """Bring a database created before a column existed up to the current schema.

    Awaited before the server takes traffic, because the sync and the cache fingerprint
    both read what it adds. Every statement is additive and idempotent (src/db/migrate.py).
    An unreachable database is /health's to report, not a reason to refuse to start.
    """
    if not settings.migrate_on_startup:
        return

    from src.db.migrate import migrate

    try:
        await run_in_threadpool(migrate)
    except Exception as error:  # noqa: BLE001
        logger.warning("schema migration skipped: %s", error)


async def _sync_loop(settings, sync: Callable) -> None:
    """Keep the mirror inside the freshness bound: one interval, then the time a sync takes.

    Sleeps first, so a restart does not re-embed anything the moment it comes up. A failed
    run is already recorded per source by sync_sources; this only keeps the loop alive.
    """
    interval = settings.sync_interval_minutes * 60
    while True:
        await asyncio.sleep(interval)
        try:
            await run_in_threadpool(sync, "schedule", None)
        except Exception:  # noqa: BLE001 - the next interval tries again
            logger.exception("scheduled sync failed")


def create_app(
    nodes: dict | None = None,
    user_loader=load_user,
    credential_loader=load_credentials,
    cache=None,
    throttle=None,
    query_limit=None,
    sync=None,
    synced_at=None,
    *,
    refusal_deadline: float | None = None,
    clock=time.monotonic,
    sleeper=asyncio.sleep,
) -> FastAPI:
    """Build the app. `clock` and `sleeper` are injectable so padding is testable
    without sleeping; `refusal_deadline` overrides the configured value, 0 disables it.
    """
    if sync is None:
        def sync(trigger: str, user_id: int | None = None):
            return sync_sources(trigger=trigger, user_id=user_id)

    if synced_at is None:
        synced_at = last_synced

    async def stamped(response: AskerResponse) -> AskerResponse:
        """Say how current each cited source is. A refusal has no citations, so nothing here
        can reach one (§5); an answer is stamped on the way out and never in the cache, so a
        hit reports the sync as it is now rather than as it was when the answer was filed."""
        if not response.citations:
            return response
        when = await run_in_threadpool(synced_at)
        return response.model_copy(update={"citations": [
            c.model_copy(update={"as_of": when.get(c.source_platform)}) for c in response.citations
        ]})

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        current = get_settings()
        await _migrate(current)
        # Not awaited: the server should be answering before the model finishes
        # loading, and a warm-up that fails is a slow first answer, not a bad start.
        tasks = [asyncio.create_task(_warm(current))]
        if current.sync_interval_minutes > 0:
            tasks.append(asyncio.create_task(_sync_loop(current, sync)))
        yield
        for task in tasks:
            task.cancel()

    app = FastAPI(title="Internal Brain", lifespan=lifespan)
    nodes = nodes or default_nodes()
    graph = build_graph(**nodes)
    # Built before the routers: both the compliance router and /query depend on it.
    caller, officer = build_dependencies(user_loader)
    app.include_router(build_router(user_loader, officer=officer, sync=sync))
    app.include_router(build_ui_router())
    settings = get_settings()
    if throttle is None:
        throttle = SlidingWindow(
            limit=settings.login_max_attempts, window=settings.login_lockout_seconds
        )
    if query_limit is None:
        query_limit = SlidingWindow(
            limit=settings.query_max_per_window, window=settings.query_window_seconds
        )
    if cache is None and settings.query_cache_enabled:
        cache = PermissionAwareCache(
            similarity_threshold=settings.query_cache_similarity,
            ttl_seconds=settings.query_cache_ttl_seconds or None,
        )
    if refusal_deadline is None:
        refusal_deadline = (
            settings.refusal_deadline_seconds if settings.refusal_padding_enabled else 0.0
        )
    # The Ask page states what refusals do with timing, and that is only true of the deadline
    # this app actually enforces, so the UI router reads it from here (src/api/ui.py).
    app.state.refusal_hold = refusal_deadline

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
    async def login(body: LoginRequest, raw: Request) -> Token:
        settings = get_settings()
        # Both counters, so guessing one account's password and spraying many accounts
        # from one client are each bounded. The email key is counted even when no such
        # account exists — a lockout that only ever fired for real addresses would
        # answer "does this account exist?", which is what the shared error message and
        # the equal-time check below exist to refuse.
        who = f"email:{body.email.strip().lower()}"
        where = f"ip:{raw.client.host if raw.client else 'unknown'}"
        wait = throttle.retry_after(who, where)
        if wait:
            raise HTTPException(
                status_code=429,
                detail="Too many sign-in attempts. Try again in a few minutes.",
                headers={"Retry-After": str(int(wait) + 1)},
            )

        def check() -> int | None:
            found = credential_loader(body.email)
            # verify_password is run even when the email is unknown, against a hash
            # that cannot match, so a wrong address and a wrong password cost the
            # same time. Otherwise the response time enumerates accounts.
            user_id, stored = found if found else (None, None)
            return user_id if verify_password(body.password, stored) else None

        user_id = await run_in_threadpool(check)
        if user_id is None:
            throttle.record(who, where)
            # One message for both, so it never says which half was wrong.
            raise HTTPException(status_code=401, detail="That email and password do not match.")
        throttle.clear(who, where)
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

        # Per caller, counted BEFORE the work and regardless of how it ends. Every
        # refusal is held to the deadline, so without this one caller can park a task
        # per request for REFUSAL_DEADLINE_SECONDS while the model queues behind them
        # — degrading the very number this product argues from.
        #
        # An answer and a refusal cost exactly the same allowance, and a rejection
        # never consults the query. Charging them differently, or exempting anything,
        # would let a caller read their own remaining allowance as a signal about
        # what they had just been told (§5).
        seat = f"user:{user.id}"
        if query_limit is not None:
            wait = query_limit.retry_after(seat)
            if wait:
                raise HTTPException(
                    status_code=429,
                    detail="Too many questions at once. Try again shortly.",
                    headers={"Retry-After": str(int(wait) + 1)},
                )
            query_limit.record(seat)

        # Async handler, blocking work on the threadpool: padding then awaits on the
        # event loop and holds no worker. A sync sleep would hold one of ~40 for the
        # full deadline, and parallel restricted questions would exhaust the pool.
        # `user` is already resolved: the dependency verified the token and read the
        # row before this handler ran.

        stamp = None
        if cache is not None:
            # Taken BEFORE the lookup and the graph, and handed to `put`: an answer is filed
            # under the state of the world it was computed in, not the one it finished in.
            # See src/cache.py for what filing it after would let through.
            stamp = await run_in_threadpool(cache.fingerprint, user)
            entry = await run_in_threadpool(
                cache.lookup, request.query, user, fingerprint=stamp
            )
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
                return await stamped(cached.model_copy(update={
                    "trace": [Stage(node="cache", hop=0, ms=(clock() - started) * 1000)]
                }))

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
        # visible in the text. Except answers resting on a RESTRICTED document: a hit never
        # reaches the query-time source recheck, so those take the full path every time
        # (found by evals/freshness_probe.py, which saw one served after its ACL narrowed).
        if cache is not None and not await run_in_threadpool(_rests_on_restricted, response):
            await run_in_threadpool(
                cache.put, request.query, user, response, state.get("route"),
                fingerprint=stamp,
            )
        return await stamped(response)

    return app


app = create_app()
