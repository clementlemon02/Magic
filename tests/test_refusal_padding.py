"""Constant-time refusal (docs/design/constant-time-refusal.md).

A permission-conflict refusal and a refusal for lack of evidence must take the same
wall-clock time, or the response is a timing oracle for whether a restricted
document exists. These use an injected clock, so nothing actually sleeps — except
the concurrency test, which has to.
"""

import asyncio
import time

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.graph.state import GENERIC_REFUSAL, PermConflict, UserContext, VerificationResult

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)
DEADLINE = 4.0


class FakeClock:
    """Time only moves when a node spends it or the app sleeps."""

    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def spend(self, seconds: float) -> None:
        self.now += seconds

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


CONFLICT = PermConflict(
    document_id=7, source_platform="confluence", source_ref="COMPLIANCE/aml-escalation",
    sensitivity="restricted", score_margin=0.31,
)


def _nodes(clock: FakeClock, **overrides):
    base = {
        "router": lambda s: {"route": "rag"},
        "retrieval": lambda s: (clock.spend(0.4), {"hop_count": s.get("hop_count", 0) + 1})[1],
        "sql_tool": lambda s: {"sql_result": None},
        "clarification": lambda s: {"clarification_question": "Which one?"},
        "synthesizer": lambda s: (clock.spend(0.4), {"draft_answer": "45 days.", "citations": []})[1],
        "verifier": lambda s: {
            "verification": VerificationResult(grounded=True, unsupported=[], confidence=0.9)
        },
        "escalation": lambda s: {"escalated": True},
        "audit": lambda s: {"audit_events": []},
    }
    base.update(overrides)
    return base


def _client(clock, **overrides) -> TestClient:
    app = create_app(
        nodes=_nodes(clock, **overrides),
        user_loader=lambda uid: ALEX,
        refusal_deadline=DEADLINE,
        clock=clock,
        sleeper=clock.sleep,
    )
    return TestClient(app, raise_server_exceptions=False)


def _ask(client, query="anything"):
    return client.post("/query", json={"query": query, "user_id": 1})


def _conflict_retrieval(clock):
    """The fast path: a restricted match short-circuits straight to Escalation."""
    return lambda s: (clock.spend(1.2), {"hop_count": 1, "permission_conflicts": [CONFLICT]})[1]


def _ungrounded_verifier():
    return lambda s: {
        "verification": VerificationResult(grounded=False, unsupported=["none"], confidence=0.9)
    }


def test_a_conflict_refusal_is_held_to_the_deadline():
    clock = FakeClock()
    r = _ask(_client(clock, retrieval=_conflict_retrieval(clock)))
    assert r.json()["text"] == GENERIC_REFUSAL
    assert clock.now == pytest.approx(DEADLINE)
    assert clock.slept == [pytest.approx(DEADLINE - 1.2)]


def test_conflict_and_hop_cap_refusals_take_identical_time():
    """The property itself. Unpadded these were separable from one measurement."""
    fast = FakeClock()
    _ask(_client(fast, retrieval=_conflict_retrieval(fast)))

    slow = FakeClock()
    _ask(_client(slow, verifier=_ungrounded_verifier()))  # three hops, then escalation

    assert fast.now == pytest.approx(DEADLINE)
    assert slow.now == pytest.approx(DEADLINE)
    assert fast.slept[0] > slow.slept[0], "the fast path should have waited longer"


def test_the_refusal_text_is_still_byte_identical():
    fast, slow = FakeClock(), FakeClock()
    a = _ask(_client(fast, retrieval=_conflict_retrieval(fast))).json()
    b = _ask(_client(slow, verifier=_ungrounded_verifier())).json()
    assert a == b


def test_an_answer_is_not_delayed():
    clock = FakeClock()
    r = _ask(_client(clock))
    assert r.json()["text"] == "45 days."
    assert clock.slept == []


def test_a_clarification_is_not_delayed():
    """Decided by the Router before retrieval, so it can't depend on restricted content."""
    clock = FakeClock()
    r = _ask(_client(clock, router=lambda s: {"route": "clarify"}))
    assert r.json()["text"] == "Which one?"
    assert clock.slept == []


def test_a_refusal_past_the_deadline_is_not_delayed_further():
    """The one-directional tail leak: reported by the eval, never made worse."""
    clock = FakeClock()
    late = lambda s: (clock.spend(5.5), {"hop_count": 1, "permission_conflicts": [CONFLICT]})[1]
    _ask(_client(clock, retrieval=late))
    assert clock.slept == []
    assert clock.now == 5.5


def test_an_error_is_not_padded():
    """An outage should report fast; a 5xx is already distinguishable."""
    clock = FakeClock()

    def down(s):
        raise psycopg.OperationalError("connection lost")

    r = _ask(_client(clock, retrieval=down))
    assert r.status_code == 503
    assert clock.slept == []


def test_the_clock_starts_before_the_caller_is_resolved():
    """Caller resolution precedes the graph; it must be inside the envelope."""
    clock = FakeClock()

    def slow_lookup(user_id):
        clock.spend(0.7)
        return ALEX

    app = create_app(
        nodes=_nodes(clock, retrieval=_conflict_retrieval(clock)),
        user_loader=slow_lookup,
        refusal_deadline=DEADLINE,
        clock=clock,
        sleeper=clock.sleep,
    )
    _ask(TestClient(app))
    assert clock.now == pytest.approx(DEADLINE)
    assert clock.slept == [pytest.approx(DEADLINE - 0.7 - 1.2)]


def test_a_zero_deadline_disables_padding():
    clock = FakeClock()
    app = create_app(
        nodes=_nodes(clock, retrieval=_conflict_retrieval(clock)),
        user_loader=lambda uid: ALEX,
        refusal_deadline=0.0,
        clock=clock,
        sleeper=clock.sleep,
    )
    _ask(TestClient(app))
    assert clock.slept == []


def test_padding_holds_no_worker_thread():
    """Real time, real sleep. The defence must not become a DoS amplifier.

    Fifty refusals are held at once. A sync sleep would pin threadpool workers for
    the whole deadline and a concurrent answer would queue behind them; an async
    sleep releases them, so the answer returns at once and the refusals all finish
    together rather than one after another.
    """
    deadline = 0.5
    nodes = {
        "router": lambda s: {"route": "rag"},
        "retrieval": lambda s: (
            {"hop_count": 1, "permission_conflicts": [CONFLICT]}
            if "restricted" in s["query"]
            else {"hop_count": 1}
        ),
        "sql_tool": lambda s: {"sql_result": None},
        "clarification": lambda s: {"clarification_question": "?"},
        "synthesizer": lambda s: {"draft_answer": "45 days.", "citations": []},
        "verifier": lambda s: {
            "verification": VerificationResult(grounded=True, unsupported=[], confidence=0.9)
        },
        "escalation": lambda s: {"escalated": True},
        "audit": lambda s: {"audit_events": []},
    }
    app = create_app(nodes=nodes, user_loader=lambda uid: ALEX, refusal_deadline=deadline)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async def timed(q):
                t = time.monotonic()
                r = await client.post("/query", json={"query": q, "user_id": 1})
                return r.json()["text"], time.monotonic() - t

            started = time.monotonic()
            refusals = [asyncio.create_task(timed("restricted question")) for _ in range(50)]
            await asyncio.sleep(0.05)  # let them all reach their padding
            answer_text, answer_time = await timed("permitted question")
            results = await asyncio.gather(*refusals)
            return answer_text, answer_time, results, time.monotonic() - started

    answer_text, answer_time, results, total = asyncio.run(run())

    assert answer_text == "45 days."
    assert answer_time < deadline / 2, f"answer waited {answer_time:.2f}s behind held refusals"
    assert all(text == GENERIC_REFUSAL for text, _ in results)
    assert all(t >= deadline for _, t in results), "a refusal escaped its padding"
    assert total < deadline * 3, f"50 refusals took {total:.2f}s — they serialised"
