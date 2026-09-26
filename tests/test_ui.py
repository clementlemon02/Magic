"""The two UI pages: the asker's at `/` and the compliance dashboard at `/dashboard`.

Both are markup only. Everything they show arrives from the API — `/query` resolves
the caller from the database, the two gap reports require the compliance role — so
these check that the shells really are inert, rather than that they look right.
"""

from fastapi.testclient import TestClient

from src.api.ui import PAGES, build_router
from src.api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_the_page_is_served_as_html():
    response = _client().get("/dashboard")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Gaps" in response.text


def test_the_page_is_not_gated_because_it_carries_no_data():
    """Deliberate: gating the shell would be a second place that has to agree who an
    officer is, and the shell discloses nothing. The DATA is gated."""
    assert _client().get("/dashboard", headers={}).status_code == 200


def test_the_page_embeds_no_report_data():
    """The shell must never be a second copy of what /access-gaps is gated to protect."""
    page = PAGES["dashboard"].read_text(encoding="utf-8")
    for restricted_ref in (
        "COMPLIANCE/aml-escalation",
        "file-aml-evidence",
        "RISK/AML-24",
        "compliance-alerts",
    ):
        assert restricted_ref not in page
    assert "fetch(" in page  # it asks for the data at runtime instead


def test_the_page_sends_the_caller_identity_header():
    """Without X-User-Id the report route cannot tell who is asking, and 403s."""
    assert "X-User-Id" in PAGES["dashboard"].read_text(encoding="utf-8")


def test_dashboard_route_is_hidden_from_the_api_schema():
    """It is a page, not an endpoint; listing it in /openapi.json only adds noise."""
    routes = [r for r in build_router().routes if getattr(r, "path", "") == "/dashboard"]
    assert routes and routes[0].include_in_schema is False


def test_the_ask_page_is_served_and_posts_to_query():
    response = _client().get("/")
    assert response.status_code == 200
    assert "<title>Ask" in response.text
    assert '"/query"' in response.text


def test_the_ask_page_embeds_no_answers():
    """It must not ship a canned answer that could be shown without asking the API."""
    page = PAGES["ask"].read_text(encoding="utf-8")
    for leak in ("SGD 9,500", "Project Nightingale", "45 days"):
        assert leak not in page


def test_the_ask_page_carries_the_refusal_string_only_to_recognise_it():
    """§5: the page styles a refusal differently, so it has to match the one string —
    but it must never author a refusal of its own."""
    page = PAGES["ask"].read_text(encoding="utf-8")
    assert page.count("I don't have an answer you're permitted") == 1
    assert "startsWith" in page
