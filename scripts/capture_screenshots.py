"""Capture README screenshots from the running app.

Scripted rather than taken by hand so the images can be regenerated after any UI
change, and so they always show real model output instead of a mock-up.

Prerequisites -- both servers running:

    cd backend && uvicorn app.main:app --port 8000
    cd frontend && npm run dev

Then::

    pip install playwright && playwright install chromium
    python scripts/capture_screenshots.py

Writes PNGs into ``docs/screenshots/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "screenshots"

MIXED_REVIEW = (
    "The display is beautiful and the camera takes excellent photos, "
    "but the battery life is disappointing."
)

SAMPLE_REVIEWS = """Camera is superb, photos look professional. Battery lasts all day too.
Battery dies after four hours. Really disappointing for the price.
Fast delivery and the packaging was excellent. Phone works great.
The screen is gorgeous but it is far too expensive for what you get.
Terrible customer service. Took three weeks to get a reply.
Great value for money. Does everything I need.
Software is buggy and it freezes constantly. Camera is decent though.
Build quality feels cheap, but the display is bright and sharp."""


def set_textarea(page, selector: str, text: str) -> None:
    """Set a controlled React textarea.

    ``fill()`` alone does not always drive React's onChange, so the value is set
    through the native setter and an input event is dispatched explicitly.
    """
    page.evaluate(
        """([selector, value]) => {
            const el = document.querySelector(selector);
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value').set;
            setter.call(el, value);
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }""",
        [selector, text],
    )


def capture(base_url: str, theme: str) -> list[Path]:
    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if theme == "light" else "-dark"
    written: list[Path] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)

        # Set the theme before the app reads it on first paint.
        page.goto(base_url, wait_until="networkidle")
        page.evaluate("t => localStorage.setItem('absa-theme', t)", theme)
        page.reload(wait_until="networkidle")

        # ---- single review -------------------------------------------------
        set_textarea(page, "textarea", MIXED_REVIEW)
        page.click("button[type=submit]")
        page.wait_for_selector(".aspect", timeout=15000)
        page.wait_for_timeout(400)

        path = OUT_DIR / f"single-review{suffix}.png"
        page.screenshot(path=str(path), full_page=True)
        written.append(path)

        # ---- product dashboard --------------------------------------------
        page.click("[role=tab]:has-text('Product dashboard')")
        page.wait_for_selector("textarea", timeout=15000)

        textareas = page.locator("textarea")
        set_textarea(page, "textarea", SAMPLE_REVIEWS)
        page.fill("input[aria-label='Product name']", "Sample Phone")
        page.click("button:has-text('Analyse reviews')")
        page.wait_for_selector(".stat", timeout=30000)
        # Recharts animates in; wait for the bars to settle before capturing.
        page.wait_for_selector(".recharts-bar-rectangle", timeout=15000)
        page.wait_for_timeout(1200)

        path = OUT_DIR / f"dashboard{suffix}.png"
        page.screenshot(path=str(path), full_page=True)
        written.append(path)

        browser.close()

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:5173")
    parser.add_argument("--themes", nargs="+", default=["light", "dark"])
    args = parser.parse_args()

    try:
        import playwright  # noqa: F401
    except ImportError:
        print(
            "playwright is not installed.\n"
            "  pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        return 1

    all_written = []
    for theme in args.themes:
        try:
            all_written += capture(args.url, theme)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to capture {theme}: {exc}", file=sys.stderr)
            print(
                "Are both servers running? "
                "(uvicorn on :8000 and npm run dev on :5173)",
                file=sys.stderr,
            )
            return 1

    for path in all_written:
        size_kb = path.stat().st_size / 1024
        print(f"  {path.relative_to(REPO_ROOT)}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
