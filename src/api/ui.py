"""The web UI: the asker's page at `/` and the compliance dashboard at `/dashboard`.

Neither page carries data. Both are inert markup that fetch the API from the
browser, so every gate stays where it already is — `/query` resolves the caller
from the database and `/access-gaps` and `/knowledge-gaps` require the compliance
role. Gating the shells as well would only be a second place that has to agree who
an officer is, and the shells disclose nothing.

No template engine: the pages are static and fill themselves from JSON, so adding
Jinja2 would buy nothing. They are read from disk rather than embedded as Python
strings so their CSS and JS braces are not fighting `.format`.
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

PAGES = {name: Path(__file__).with_name(f"{name}.html") for name in ("ask", "dashboard")}


def build_router() -> APIRouter:
    router = APIRouter()

    def page(name: str) -> HTMLResponse:
        # Read per request: a demo is edited and reloaded, not restarted.
        return HTMLResponse(PAGES[name].read_text(encoding="utf-8"))

    @router.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ask() -> HTMLResponse:
        return page("ask")

    @router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> HTMLResponse:
        return page("dashboard")

    return router
