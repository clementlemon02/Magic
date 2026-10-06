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

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

PAGES = {name: Path(__file__).with_name(f"{name}.html") for name in ("ask", "dashboard", "audit", "sources")}
STYLESHEET = Path(__file__).with_name("static") / "app.css"
SCRIPT = Path(__file__).with_name("static") / "app.js"


def build_router() -> APIRouter:
    router = APIRouter()

    def page(name: str, request: Request) -> HTMLResponse:
        # Read per request: a demo is edited and reloaded, not restarted.
        html = PAGES[name].read_text(encoding="utf-8")
        # The Ask page states what refusals do with timing, which is only true of the server
        # it is served from, so create_app records the deadline it enforces (0 = no hold) and
        # the page is told it rather than hardcoding one a setting can turn off. Not a secret:
        # a stopwatch measures it, and the defence never relied on it. An app that never set it
        # (this router built on its own) claims no hold, the safe direction to be wrong in.
        hold = getattr(request.app.state, "refusal_hold", 0.0)
        return HTMLResponse(html.replace("__REFUSAL_HOLD_SECONDS__", f"{hold:g}"))

    @router.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ask(request: Request) -> HTMLResponse:
        return page("ask", request)

    @router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard(request: Request) -> HTMLResponse:
        return page("dashboard", request)

    @router.get("/audit", response_class=HTMLResponse, include_in_schema=False)
    def audit(request: Request) -> HTMLResponse:
        # The page is inert like the others; /audit/recent and /audit/{id} are the gates.
        return page("audit", request)

    @router.get("/sources", response_class=HTMLResponse, include_in_schema=False)
    def sources(request: Request) -> HTMLResponse:
        return page("sources", request)

    @router.get("/ui/app.js", include_in_schema=False)
    def script() -> Response:
        return Response(SCRIPT.read_text(encoding="utf-8"), media_type="text/javascript")

    @router.get("/ui/app.css", include_in_schema=False)
    def stylesheet() -> Response:
        # One stylesheet for both pages, so they cannot drift apart. Served from a
        # route rather than a StaticFiles mount: it is one file, and a mount would
        # expose whatever else ends up in that directory.
        return Response(STYLESHEET.read_text(encoding="utf-8"), media_type="text/css")

    return router
