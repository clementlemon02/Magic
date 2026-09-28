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
    assert "<title>Internal Brain" in response.text
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


def test_the_stylesheet_is_served_and_shared_by_both_pages():
    """One stylesheet, so the two pages cannot drift apart visually."""
    client = _client()
    css = client.get("/ui/app.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert "--withheld" in css.text  # semantic tokens, not just the accent
    for name in ("ask", "dashboard"):
        assert '/ui/app.css' in PAGES[name].read_text(encoding="utf-8")


def test_both_pages_define_the_full_palette_on_bare_root():
    """A token defined only inside a media query renders one theme's text on the
    other theme's background. Every colour has to exist before either override."""
    css = PAGES["ask"].read_text(encoding="utf-8")  # pages link it; read the file itself
    from src.api.ui import STYLESHEET

    sheet = STYLESHEET.read_text(encoding="utf-8")
    base = sheet[sheet.index(":root {"): sheet.index("@media (prefers-color-scheme: dark)")]
    for token in ("--ground", "--surface", "--ink", "--muted", "--line", "--accent",
                  "--withheld", "--caution", "--clear"):
        assert token in base, f"{token} is not defined on bare :root"
    assert 'data-theme="dark"' in sheet and "prefers-color-scheme: dark" in sheet


def test_hidden_elements_stay_hidden():
    """`.readout{display:flex}` beats the UA rule for [hidden]; both pages use the
    attribute to hide the live readouts until there is something to show."""
    from src.api.ui import STYLESHEET

    assert "[hidden] { display: none !important; }" in STYLESHEET.read_text(encoding="utf-8")


def test_the_deadline_claim_is_bounded_by_the_measured_time():
    """A refusal that ran long escaped the deadline. Claiming it was held, next to a
    number that says otherwise, discredits the guarantee the product is built on."""
    page = PAGES["ask"].read_text(encoding="utf-8")
    assert "seconds >= 3.9 && seconds <= 4.6" in page
    assert "ran past the 4.0s deadline" in page


def test_the_audit_page_is_served_and_carries_no_trail_of_its_own():
    client = _client()
    response = client.get("/audit")
    assert response.status_code == 200
    assert "<title>Audit trail" in response.text
    page = PAGES["audit"].read_text(encoding="utf-8")
    # Everything it shows arrives from the two gated routes at runtime.
    assert "/audit/recent" in page and "fetch(" in page
    for leak in ("COMPLIANCE/aml-escalation", "Withheld:", "file-aml-evidence"):
        assert leak not in page


def test_small_labels_meet_contrast():
    """#8b98a2 on white is ~2.9:1 — it failed AA for the 10.5px uppercase labels."""
    from src.api.ui import STYLESHEET

    assert "#8b98a2" not in STYLESHEET.read_text(encoding="utf-8")


def test_a_refusal_can_carry_no_evidence():
    """§5 structurally: build_response gives a refusal GENERIC_REFUSAL and nothing
    else, so `citations` is empty and neither passage nor query has a path out."""
    from src.graph.graph import build_response
    from src.graph.state import GENERIC_REFUSAL

    reply = build_response({"escalated": True, "final_answer": "leak", "citations": ["x"]})
    assert reply.text == GENERIC_REFUSAL
    assert reply.citations == []


def test_the_ask_page_escapes_a_query_before_marking_it():
    """Marking the ACL line before escaping would put the span through esc() and
    print the tag as text."""
    page = PAGES["ask"].read_text(encoding="utf-8")
    assert "esc(sql).split" in page


def test_every_page_carries_the_same_navigation():
    """Four pages built one at a time drifted into four different navs: the gaps page
    had neither Audit nor Sources, so a reader could reach them only by typing a URL."""
    import re

    navs = {}
    for name, page in PAGES.items():
        nav = re.search(r"<nav>(.*?)</nav>", page.read_text(encoding="utf-8"), re.S).group(1)
        navs[name] = sorted(re.findall(r'href="([^"]+)"', nav))
    expected = ["/", "/audit", "/dashboard", "/sources"]
    for name, links in navs.items():
        assert links == expected, f"{name} links to {links}"


def test_each_page_marks_itself_as_the_current_one():
    import re

    for name, path in (("ask", "/"), ("dashboard", "/dashboard"),
                       ("audit", "/audit"), ("sources", "/sources")):
        nav = re.search(r"<nav>(.*?)</nav>", PAGES[name].read_text(encoding="utf-8"), re.S).group(1)
        assert f'href="{path}" aria-current="page"' in nav, name


def _rule(css: str, selector: str) -> str:
    """One declaration block. Slicing a fixed number of characters instead runs into
    whatever rule follows — which is how the first version of this test 'found' a
    border on .panel that belonged to the inputs below it."""
    start = css.index(selector + " {")
    return css[start: css.index("}", start)]


def test_the_radius_scale_is_three_tokens_not_a_value_per_component():
    """Eight literal radii were the most generated-looking thing in the sheet: a
    system nobody decided. Components take a token now."""
    import re
    from src.api.ui import STYLESHEET

    css = STYLESHEET.read_text(encoding="utf-8")
    assert set(re.findall(r"--r-(control|surface|pill):", css)) == {"control", "surface", "pill"}
    # A circle is not a corner, and an explicit 0 is a decision rather than a stray value.
    literals = {v for v in re.findall(r"border-radius: ([^;]+);", css) if "var(--r-" not in v}
    assert literals <= {"50%", "0"}, f"literal radii left: {sorted(literals)}"


def test_only_what_earns_an_edge_keeps_one():
    """Nine bordered cards competed and nothing led. The answer and the officer-only
    explanation keep a border; a control panel and a list row do not."""
    from src.api.ui import STYLESHEET

    css = STYLESHEET.read_text(encoding="utf-8")
    assert "border: 1px solid" not in _rule(css, ".panel"), "a control panel is chrome"
    assert "border: 0" in _rule(css, ".request"), "a worklist is a list, not cards"
    assert "border: 1px solid var(--line)" in _rule(css, ".answer")
    assert "border: 1px solid var(--line)" in _rule(css, ".warded")


def test_the_comparison_runs_the_two_callers_one_at_a_time():
    """One model serves one request at a time, so two in flight would queue and the
    second clock would show the wait rather than the work — a lie on the one screen
    whose entire point is timing."""
    page = PAGES["ask"].read_text(encoding="utf-8")
    assert "Sequential, not parallel" in page
    assert "Promise.all" not in page, "the two callers must not be raced"


def test_the_comparison_draws_the_deadline_to_scale():
    """A refusal landing on the deadline and an answer running past it is the whole
    property, read without a number. It has to be plotted, not asserted."""
    page = PAGES["ask"].read_text(encoding="utf-8")
    assert "const DEADLINE = 4.0;" in page and "const SCALE = 8.0;" in page
    assert "(DEADLINE / SCALE) * 100" in page, "the marker must sit at its true position"
