"""The access-gap dashboard — the compliance officer's view of §4's Access-Gap report.

The page itself carries NO data and is deliberately not gated: it is inert markup
that fetches `/access-gaps` in the browser, and that route is officer-gated as
before. Serving the shell openly is not a disclosure, and gating it would only mean
a second place that has to agree about who an officer is.

No template engine: the page is static and fills itself from JSON, so adding Jinja2
would buy nothing. It is read from disk rather than embedded as a Python string so
the CSS and JS braces are not fighting `.format`.
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

PAGE = Path(__file__).with_name("dashboard.html")


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> HTMLResponse:
        # Read per request: a demo is edited and reloaded, not restarted.
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))

    return router
