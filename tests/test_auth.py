"""Authentication: who is asking, and only that.

The property these guard is the one §1 states for retrieval — a caller cannot assert
what they may see. A token carries a subject; role and clearance are read from the
database on every request.
"""

import time

import pytest
from fastapi.testclient import TestClient

from src.api.auth import hash_password, issue_token, read_token, verify_password
from src.api.main import create_app
from src.graph.state import UserContext
from tests.helpers import as_user

ALEX = UserContext(id=1, role="support", dept="support", clearance_level=0)
MARCUS = UserContext(id=2, role="compliance", dept="compliance", clearance_level=1)


def _app(user=ALEX, credentials=None):
    def nodes():
        from src.api.main import default_nodes

        return default_nodes()

    return TestClient(
        create_app(
            nodes={k: (lambda s: {"route": "rag"}) for k in
                   ["router", "retrieval", "sql_tool", "clarification", "synthesizer",
                    "verifier", "escalation", "audit"]},
            user_loader=lambda uid: user,
            credential_loader=credentials or (lambda email: None),
        )
    )


# --- passwords ---------------------------------------------------------------

def test_a_password_verifies_against_its_own_hash():
    stored = hash_password("correct horse")
    assert verify_password("correct horse", stored)
    assert not verify_password("Correct Horse", stored)


def test_two_people_with_the_same_password_do_not_share_a_hash():
    """A per-password salt: otherwise one cracked hash cracks every account using it."""
    assert hash_password("demo") != hash_password("demo")


def test_an_account_with_no_password_cannot_sign_in():
    """False rather than an error, so a missing hash and a wrong password look alike."""
    assert not verify_password("anything", None)
    assert not verify_password("anything", "")
    assert not verify_password("anything", "not-a-scrypt-hash")


# --- tokens ------------------------------------------------------------------

def test_a_token_names_its_subject():
    assert read_token(issue_token(7)) == 7


def test_an_expired_token_is_refused():
    assert read_token(issue_token(7, now=0)) is None


def test_a_token_signed_with_another_key_is_refused(monkeypatch):
    from src.config import get_settings

    forged = issue_token(7)
    monkeypatch.setenv("AUTH_SECRET", "a-different-key")
    get_settings.cache_clear()
    try:
        assert read_token(forged) is None
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("token", ["", "x", "a.b.c", "....", "a.b"])
def test_a_malformed_token_is_refused_rather_than_raising(token):
    assert read_token(token) is None


def test_the_payload_cannot_be_edited_without_breaking_the_signature():
    """The whole point: a caller who could rewrite `sub` could be anyone."""
    header, payload, signature = issue_token(1).split(".")
    other = issue_token(2).split(".")[1]
    assert read_token(f"{header}.{other}.{signature}") is None


def test_a_token_carries_no_role():
    """A role in the token would be a claim the caller controls, and every ACL
    predicate downstream is built from it."""
    import base64
    import json

    payload = issue_token(2).split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    assert set(claims) == {"sub", "exp"}
    assert "role" not in claims and "clearance_level" not in claims


# --- the API -----------------------------------------------------------------

def test_query_without_a_token_is_refused():
    assert _app().post("/query", json={"query": "anything"}).status_code == 401


def test_query_with_a_token_reaches_the_graph():
    assert _app().post("/query", json={"query": "x"}, headers=as_user(1)).status_code != 401


def test_a_wrong_password_and_an_unknown_email_are_the_same_answer():
    """Different answers here enumerate accounts."""
    stored = hash_password("demo")
    known = _app(credentials=lambda email: (1, stored)).post(
        "/auth/login", json={"email": "alex@x.example", "password": "wrong"})
    unknown = _app(credentials=lambda email: None).post(
        "/auth/login", json={"email": "nobody@x.example", "password": "demo"})
    assert known.status_code == unknown.status_code == 401
    assert known.json() == unknown.json()


def test_signing_in_returns_a_token_that_names_the_right_person():
    stored = hash_password("demo")
    client = _app(user=MARCUS, credentials=lambda email: (2, stored))
    body = client.post("/auth/login", json={"email": "m@x.example", "password": "demo"}).json()
    assert read_token(body["access_token"]) == 2
    assert body["token_type"] == "bearer" and body["expires_in"] > 0


def test_the_role_comes_from_the_database_not_the_token():
    """A token for user 2 is answered with whatever the database says user 2 is —
    so re-signing a token cannot promote anyone."""
    client = _app(user=ALEX)  # the loader answers `support` whoever is asked for
    me = client.get("/auth/me", headers=as_user(2)).json()
    assert me["role"] == "support" and me["clearance_level"] == 0


# --- sign-in throttling ---------------------------------------------------------

def _throttled_app(**kw):
    """An app whose only account is Alex, with a tiny allowance."""
    from src.api.auth import SlidingWindow, hash_password

    stored = hash_password("demo")
    return create_app(
        user_loader=lambda uid: ALEX,
        credential_loader=lambda email: (1, stored) if email == "alex@x.example" else None,
        throttle=SlidingWindow(limit=3, window=600.0, **kw),
    )


def _try(client, email, password):
    return client.post("/auth/login", json={"email": email, "password": password})


def test_repeated_wrong_passwords_are_locked_out():
    client = TestClient(_throttled_app())
    for _ in range(3):
        assert _try(client, "alex@x.example", "nope").status_code == 401
    locked = _try(client, "alex@x.example", "nope")
    assert locked.status_code == 429
    assert "Retry-After" in locked.headers
    # Even the RIGHT password waits: otherwise the lockout is a free oracle for
    # whether the guess that triggered it was close.
    assert _try(client, "alex@x.example", "demo").status_code == 429


def test_an_address_that_does_not_exist_is_locked_out_the_same_way():
    """The enumeration property. A lockout that only ever fired for real accounts
    would answer "does this account exist?" from its own behaviour — which is exactly
    what the shared error message and the equal-time password check refuse to do."""
    client = TestClient(_throttled_app())
    for _ in range(3):
        assert _try(client, "ghost@x.example", "nope").status_code == 401
    real = _try(client, "alex@x.example", "nope")
    ghost = _try(client, "ghost@x.example", "nope")
    assert ghost.status_code == 429
    # The real address is not locked by the ghost's own counter, but the shared client
    # counter catches both — so neither status distinguishes them.
    assert real.status_code == ghost.status_code


def test_a_correct_password_clears_the_counter():
    """One typo must not follow you for five minutes."""
    client = TestClient(_throttled_app())
    for _ in range(2):
        assert _try(client, "alex@x.example", "nope").status_code == 401
    assert _try(client, "alex@x.example", "demo").status_code == 200
    for _ in range(3):
        assert _try(client, "alex@x.example", "nope").status_code == 401


def test_the_window_expires():
    now = [0.0]
    from src.api.auth import SlidingWindow

    t = SlidingWindow(limit=2, window=60.0, clock=lambda: now[0])
    t.record("a"), t.record("a")
    assert t.retry_after("a") > 0
    now[0] = 61.0
    assert t.retry_after("a") == 0.0
