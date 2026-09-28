"""The web UI: the asker's page at `/`, and the compliance pages at `/dashboard`
and `/audit`.

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
from fastapi.responses import HTMLResponse, Response

PAGES = {name: Path(__file__).with_name(f"{name}.html") for name in ("ask", "dashboard", "audit", "sources")}
STYLESHEET = Path(__file__).with_name("static") / "app.css"


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

    @router.get("/audit", response_class=HTMLResponse, include_in_schema=False)
    def audit() -> HTMLResponse:
        # The page is inert like the others; /audit/recent and /audit/{id} are the gates.
        return page("audit")

    @router.get("/sources", response_class=HTMLResponse, include_in_schema=False)
    def sources() -> HTMLResponse:
        return page("sources")

    @router.get("/ui/app.css", include_in_schema=False)
    def stylesheet() -> Response:
        # One stylesheet for both pages, so they cannot drift apart. Served from a
        # route rather than a StaticFiles mount: it is one file, and a mount would
        # expose whatever else ends up in that directory.
        return Response(STYLESHEET.read_text(encoding="utf-8"), media_type="text/css")

    return router
