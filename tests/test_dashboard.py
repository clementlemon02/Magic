"""The access-gap dashboard page.

The page is markup only. Everything it shows arrives from `/access-gaps`, which is
officer-gated — so these check that the shell really is inert, rather than that it
looks right.
"""

from fastapi.testclient import TestClient

from src.api.dashboard import PAGE, build_router
from src.api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_the_page_is_served_as_html():
    response = _client().get("/dashboard")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Access gaps" in response.text


def test_the_page_is_not_gated_because_it_carries_no_data():
    """Deliberate: gating the shell would be a second place that has to agree who an
    officer is, and the shell discloses nothing. The DATA is gated."""
    assert _client().get("/dashboard", headers={}).status_code == 200


def test_the_page_embeds_no_report_data():
    """The shell must never be a second copy of what /access-gaps is gated to protect."""
    page = PAGE.read_text(encoding="utf-8")
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
    assert "X-User-Id" in PAGE.read_text(encoding="utf-8")


def test_dashboard_route_is_hidden_from_the_api_schema():
    """It is a page, not an endpoint; listing it in /openapi.json only adds noise."""
    routes = [r for r in build_router().routes if getattr(r, "path", "") == "/dashboard"]
    assert routes and routes[0].include_in_schema is False
