"""Authentication — who is asking. Not what they may see.

That split is the whole point, and it is the same rule §1 states for retrieval. A
bearer token carries ONE claim, the user id. Role, department and clearance are read
from the database on every request, exactly as before, so a token cannot assert a
role and walk past the ACL predicate. Anything that put `role` in the token would
hand the caller the thing the product exists to withhold.

Stdlib only. `hashlib.scrypt` for passwords and `hmac` for the token signature are
the library's own primitives used as intended — not hand-rolled cryptography — and
adding PyJWT to sign one claim would buy nothing this does not already do. The token
is a bearer token in the JWT shape (header.payload.signature, base64url, HS256) so it
reads the way a reviewer expects.

Demo scope, stated plainly rather than implied: there is no refresh, no revocation
list and no rotation. A token is valid until it expires, and `AUTH_TOKEN_TTL_MINUTES`
is what bounds the damage. Put a real identity provider in front of this before it
leaves a demo — the dependency below is the seam to swap.
"""

import base64
import hashlib
import hmac
import json
import secrets
import time

from fastapi import Depends, Header, HTTPException

from src.config import get_settings
from src.graph.state import UserContext

_SCRYPT = {"n": 2**14, "r": 8, "p": 1}
COMPLIANCE_ROLE = "compliance"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# --- passwords ---------------------------------------------------------------


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """`scrypt$<salt>$<hash>`. A per-password salt, so two people choosing the same
    password do not share a hash."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return f"scrypt${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str | None) -> bool:
    """Constant-time, and False for a user with no password rather than an error —
    a caller must not learn which accounts exist by how the failure differs."""
    if not stored or not stored.startswith("scrypt$"):
        return False
    try:
        _, salt, expected = stored.split("$")
        digest = hashlib.scrypt(
            password.encode(), salt=_unb64(salt), dklen=32, **_SCRYPT
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, _unb64(expected))


# --- tokens ------------------------------------------------------------------


def _sign(message: bytes) -> str:
    secret = get_settings().auth_secret.encode()
    return _b64(hmac.new(secret, message, hashlib.sha256).digest())


def issue_token(user_id: int, *, now: float | None = None) -> str:
    """One claim: the subject. Deliberately no role — see the module docstring."""
    ttl = get_settings().auth_token_ttl_minutes * 60
    # `now if now is not None`, never `now or`: a caller passing 0 means the epoch,
    # and `or` reads that as "not supplied" and quietly uses the clock instead.
    issued = time.time() if now is None else now
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64(
        json.dumps({"sub": user_id, "exp": int(issued + ttl)}, separators=(",", ":")).encode()
    )
    body = f"{header}.{payload}"
    return f"{body}.{_sign(body.encode())}"


def read_token(token: str, *, now: float | None = None) -> int | None:
    """The user id a valid token names, or None. Never raises: every malformed,
    re-signed or expired token is the same 'no' to the caller."""
    try:
        header, payload, signature = token.split(".")
    except ValueError:
        return None
    if not hmac.compare_digest(signature, _sign(f"{header}.{payload}".encode())):
        return None
    try:
        claims = json.loads(_unb64(payload))
        subject, expires = int(claims["sub"]), float(claims["exp"])
    except (ValueError, TypeError, KeyError):
        return None
    return subject if (time.time() if now is None else now) < expires else None


# --- dependencies ------------------------------------------------------------


def build_dependencies(user_loader):
    """`caller` and `officer`, closed over how users are read.

    Swapping this for a real identity provider means replacing `caller`: everything
    downstream takes a UserContext and does not care where it came from.
    """

    def caller(authorization: str = Header(default="")) -> UserContext:
        scheme, _, token = authorization.partition(" ")
        user_id = read_token(token) if scheme.lower() == "bearer" and token else None
        # One answer for a missing, malformed, expired or re-signed token, and for a
        # token naming a user who no longer exists: none of them says which.
        user = user_loader(user_id) if user_id is not None else None
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Sign in first.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    def officer(user: UserContext = Depends(caller)) -> UserContext:
        # The role comes from the database row, never from the token.
        if user.role != COMPLIANCE_ROLE:
            raise HTTPException(status_code=403, detail="Compliance role required")
        return user

    return caller, officer


# --- throttling failed sign-ins -------------------------------------------------
# /auth/login had no limit at all: a caller could guess passwords as fast as the
# server would answer, and the only brake was scrypt's own cost — which is not a
# defence so much as a way to spend the server's CPU on the attacker's behalf.
#
# Counted per email AND per client, and the email counter runs for addresses that do
# not exist too. Locking only the real ones would answer "does this account exist?"
# from the lockout alone, which is the enumeration `login` is already careful to
# avoid in its timing and its error message.

class LoginThrottle:
    """In-memory failure counter with a fixed lockout window.

    ponytail: per-process, so N workers means N times the allowance and a restart
    clears it. Correct for a single-process demo; a shared counter (Redis, or a
    `login_attempts` table) is the upgrade when this runs on more than one.
    """

    def __init__(self, *, limit: int = 5, window: float = 300.0, clock=time.monotonic):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._failures: dict[str, list[float]] = {}

    def _recent(self, key: str) -> list[float]:
        now = self._clock()
        kept = [t for t in self._failures.get(key, []) if now - t < self.window]
        if kept:
            self._failures[key] = kept
        else:
            self._failures.pop(key, None)
        return kept

    def locked(self, *keys: str) -> float:
        """Seconds until the earliest key is free again, or 0.0 when none are locked."""
        waits = [
            self.window - (self._clock() - self._recent(k)[0])
            for k in keys
            if len(self._recent(k)) >= self.limit
        ]
        return max(0.0, max(waits)) if waits else 0.0

    def failed(self, *keys: str) -> None:
        now = self._clock()
        for key in keys:
            self._failures.setdefault(key, []).append(now)

    def passed(self, *keys: str) -> None:
        """A correct password clears the counters, so one typo does not follow you."""
        for key in keys:
            self._failures.pop(key, None)
