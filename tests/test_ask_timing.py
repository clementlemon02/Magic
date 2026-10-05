"""The Ask page says only what the server is actually doing about timing.

The refusal hold is a switch (REFUSAL_PADDING_ENABLED). A page that hardcoded "held to the 4.0s
deadline" told every visitor to a server with the hold off that their refusal's timing was
defended when it was not, which is the one claim this product cannot afford to get wrong.
"""

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.ui import PAGES


def _served_ask(deadline):
    return TestClient(create_app(refusal_deadline=deadline)).get("/").text


def test_the_ask_page_is_told_the_hold_it_is_served_with():
    """The hold is a switch. A page that hardcoded "held to the 4.0s deadline" told every
    visitor to a server with the hold OFF that their refusal's timing was defended when it
    was not, which is the one claim this product cannot afford to get wrong."""
    assert 'parseFloat("4")' in _served_ask(4.0)
    assert 'parseFloat("0")' in _served_ask(0.0)
    assert 'parseFloat("2.5")' in _served_ask(2.5)


def test_no_placeholder_ever_reaches_a_browser():
    for deadline in (4.0, 0.0):
        assert "__REFUSAL_HOLD_SECONDS__" not in _served_ask(deadline)


def test_an_unsubstituted_page_claims_no_hold():
    """Opened straight from disk, the placeholder is a string, which parses to NaN, which
    is falsy: the page falls back to "no hold" — the safe direction to be wrong in."""
    page = PAGES["ask"].read_text(encoding="utf-8")
    assert 'parseFloat("__REFUSAL_HOLD_SECONDS__") || 0' in page


def test_the_ask_page_hardcodes_no_deadline_and_no_unmeasured_timing():
    """Two numbers lived in this page's copy — a 4.0s deadline and "about 0.9s / 4.3s" for
    an unpadded refusal. The first is a setting. The second matched nothing in the stored
    baseline (permission refusals ~1.8s, unanswerable ones 1.3 to 2.7s by hop count)."""
    page = PAGES["ask"].read_text(encoding="utf-8")
    for stale in ("4.0s", "0.9s", "4.3s", "DEADLINE"):
        assert stale not in page, stale


def test_with_no_hold_the_page_says_so_and_draws_no_deadline():
    page = PAGES["ask"].read_text(encoding="utf-8")
    assert "Timing is not defended on this server" in page
    assert "HOLD > 0 ? `<div class=\"deadline\"" in page, "the marker must only exist when there is a deadline"
    assert "REFUSAL_PADDING_ENABLED" in page
