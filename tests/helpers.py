"""Test helpers for the authenticated API.

`/query` and every officer route take a signed bearer token now, so a test that
wants to be somebody asks for their token rather than asserting an id. That is the
same path a browser takes; nothing here bypasses verification.
"""

from src.api.auth import issue_token


def as_user(user_id: int) -> dict[str, str]:
    """Headers that authenticate as `user_id`, signed with the suite's AUTH_SECRET."""
    return {"Authorization": f"Bearer {issue_token(user_id)}"}
