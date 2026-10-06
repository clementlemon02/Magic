"""Accessibility guards for the four pages and the shared stylesheet. No browser.

These pin what an axe-core run, a contrast sweep and a phone-width check found and fixed
(docs in CLAUDE.md, "UI accessibility"). They cannot replace running those: they exist so the
specific ways this UI broke cannot quietly come back. Each test says what it was.
"""

import re

from src.api.ui import PAGES, SCRIPT, STYLESHEET

CSS = STYLESHEET.read_text(encoding="utf-8")
JS = SCRIPT.read_text(encoding="utf-8")
HTML = {name: page.read_text(encoding="utf-8") for name, page in PAGES.items()}


# --- contrast -------------------------------------------------------------------------------

def _palettes() -> dict[str, dict[str, str]]:
    """The light palette, and the dark one (the media-query block and the explicit toggle must agree)."""
    def tokens(block: str) -> dict[str, str]:
        return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", block))

    light = tokens(re.search(r"^:root\s*\{(.*?)\n\}", CSS, re.S | re.M).group(1))
    dark_media = tokens(re.search(
        r'@media \(prefers-color-scheme: dark\) \{\s*:root:not\(\[data-theme="light"\]\) \{(.*?)\n  \}',
        CSS, re.S).group(1))
    dark_toggle = tokens(re.search(r':root\[data-theme="dark"\]\s*\{(.*?)\n\}', CSS, re.S).group(1))
    assert dark_media == dark_toggle, "the two dark palettes drifted apart"
    return {"light": light, "dark": dark_toggle}


def _luminance(hex_colour: str) -> float:
    def channel(v: float) -> float:
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def test_every_text_token_reaches_4_5_on_every_surface_in_both_themes():
    """WCAG 1.4.3. `--faint` was 4.0:1 on the dark surface and 3.6:1 on the raised one, so every
    unit, label, column header and 'not used' stage was under AA in dark mode, on every page."""
    failures = []
    for theme, palette in _palettes().items():
        for text in ("ink", "muted", "faint", "accent", "withheld", "caution", "clear"):
            for surface in ("ground", "surface", "raised"):
                ratio = _contrast(palette[text], palette[surface])
                if ratio < 4.5:
                    failures.append(f"{theme}: --{text} on --{surface} is {ratio:.2f}:1")
        ratio = _contrast(palette["accent-ink"], palette["accent"])
        if ratio < 4.5:
            failures.append(f"{theme}: button text on --accent is {ratio:.2f}:1")
    assert not failures, "\n".join(failures)


def test_text_is_never_dimmed_with_opacity():
    """Opacity multiplies the contrast away: a revoked grant was 2.2:1, a locked nav link 2.9:1,
    a 'not used' pipeline stage worse. A state is said with a colour that passes, plus words."""
    rules = re.findall(r"([^{}@]+)\{([^{}]*)\}", CSS)
    dimmed = [
        selector.strip().splitlines()[-1]
        for selector, body in rules
        for value in re.findall(r"(?<![\w-])opacity:\s*([0-9.]+)", body)
        if float(value) < 1 and re.search(r"\.gone|\.locked|\.persona|\.stage", selector)
    ]
    assert not dimmed, f"text dimmed by opacity in: {dimmed}"


# --- structure and keyboard -----------------------------------------------------------------

def test_every_page_opens_with_a_skip_link_to_a_focusable_main():
    for name, html in HTML.items():
        body = html[html.index("<body>"):]
        assert re.match(r'<body>\s*<a class="skip" href="#main">Skip to content</a>', body), name
        assert re.search(r'<main [^>]*id="main"[^>]*tabindex="-1"', html), name


def test_segmented_controls_are_toggle_buttons_not_a_half_built_tab_pattern():
    """role=tab promises arrow keys, a roving tabindex and a tabpanel; these had none of them, so a
    screen reader announced a widget the keyboard could not operate. Toggle buttons say what they are."""
    for name, html in HTML.items():
        assert 'role="tab' not in html and "aria-selected" not in html, name
    assert "aria-selected" not in CSS and "aria-selected" not in JS
    assert 'aria-pressed' in HTML["audit"] and 'aria-pressed' in HTML["ask"] and 'aria-pressed' in HTML["dashboard"]


def test_a_sortable_column_is_a_button_the_keyboard_can_reach():
    """`th.sortable` carried a click handler and nothing else: unfocusable, so sorting was a mouse feature."""
    page = HTML["dashboard"]
    assert 'class="sortable"' not in page and "th.sortable" not in page
    assert '<button class="sort"' in page and 'aria-sort="descending"' in page


def test_disclosures_say_whether_they_are_open():
    assert 'aria-expanded="false" aria-controls="${eid}"' in HTML["ask"]
    assert 'peek.setAttribute("aria-expanded"' in HTML["ask"]
    assert 'aria-expanded="${open}"' in HTML["dashboard"]


def test_a_control_that_redraws_itself_hands_focus_back():
    """Each of these replaced its own button with innerHTML, so the next Tab started from the top
    of the page. A keyboard user lost their place on every sort, expand, sync and row pick."""
    assert 'refocus("button.sort"' in HTML["dashboard"] and 'refocus(".row-toggle"' in HTML["dashboard"]
    assert """.querySelector('[aria-current="true"]')?.focus()""" in HTML["audit"]
    assert 'aria-disabled' in HTML["sources"], "a disabled button drops focus; the sync button must not"
    assert ".find((b) => b.dataset.platform === button.dataset.platform" in HTML["sources"]


def test_the_sync_button_lives_outside_the_column_that_is_redrawn():
    page = HTML["sources"]
    before_sources = page[:page.index('<div id="sources"></div>')]
    assert 'id="sync"' in before_sources and 'id="sync-note" role="status"' in before_sources


def test_each_revoke_and_grant_says_which_grant():
    """Twelve buttons that all read "Revoke" are not usable by ear."""
    page = HTML["sources"]
    assert 'aria-label="${live ? "Revoke" : "Grant"} ${esc(g.source_platform)}:${esc(g.source_ref)}"' in page
    assert "labelOf(armed, false)" in page and "labelOf(button, true)" in page


def test_failures_are_announced_not_just_drawn():
    assert 'id="grant-error" class="signin-error" role="alert"' in HTML["sources"]
    assert 'id="ib-error" class="signin-error" role="alert"' in JS
    assert 'role="status"' in HTML["audit"] and 'id="announce"' in HTML["audit"]
    assert 'id="announce" role="status"' in HTML["dashboard"]


def test_a_table_redrawn_in_a_live_region_is_not_read_out_every_time():
    """The Gaps report was aria-live: every refresh read the whole table aloud. It announces the result."""
    assert 'id="out" aria-live' not in HTML["dashboard"]
    assert 'announce(`${rows.length}' in HTML["dashboard"]


def test_locked_links_say_why_in_words():
    """A lock icon and a tooltip are not read to a keyboard or a screen reader."""
    assert 'class="sr"> — compliance officers only' in JS


def test_every_form_control_has_a_name_that_is_not_its_placeholder():
    for name, html in HTML.items():
        for control in re.findall(r"<(?:input|textarea|select)\b[^>]*>", html):
            ident = re.search(r'id="([^"]+)"', control)
            labelled = (
                'aria-label="' in control
                or (ident and re.search(rf'<label[^>]*for="{re.escape(ident.group(1))}"', html))
                or 'type="hidden"' in control
            )
            assert labelled, f"{name}: unlabelled control {control[:80]}"


def test_no_page_registers_a_global_single_key_shortcut():
    """WCAG 2.1.4: "/" focused the question box from anywhere on the page, with no way to turn it off."""
    for name, html in HTML.items():
        assert 'e.key === "/"' not in html, name


# --- reflow ---------------------------------------------------------------------------------

def test_the_app_bar_wraps_instead_of_running_off_the_right_edge():
    """WCAG 1.4.10. One fixed-height row: at 375px the nav and theme switch ran off the edge and the
    page scrolled sideways to 721px."""
    rule = re.search(r"@media \(max-width: 720px\) \{\s*\.bar \{([^}]*)\}", CSS)
    assert rule and "flex-wrap: wrap" in rule.group(1) and "height: auto" in rule.group(1)


def test_no_inline_width_forces_a_column_wider_than_a_phone():
    """The grants column was `width:420px`, so the Sources page scrolled sideways at 375px."""
    for name, html in HTML.items():
        for width in re.findall(r"style=\"[^\"]*?(?<![-\w])width:\s*([^;\"]+)", html):
            match = re.match(r"(\d+)px", width.strip())
            assert not match or int(match.group(1)) < 320 or "min(" in width, f"{name}: width:{width}"


def test_focus_is_visible_on_links_and_summaries_as_well_as_controls():
    assert re.search(r"a:focus-visible", CSS) and re.search(r"summary:focus-visible", CSS)
