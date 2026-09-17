"""One-page diagnostic using default Chromium, without retries or interactions.

Setup: python -m pip install -r requirements.txt
       python -m playwright install chromium
Run:   python scripts/probe_utas_browser.py
"""

import sys

from playwright.sync_api import Error, TimeoutError, sync_playwright

PROJECT_URL = "https://www.utas.edu.au/research/degrees/available-projects?id=12167"
TIMEOUT_MS = 30_000
EXPECTED_PHRASES = (
    "Degree type",
    "Student type",
    "Scholarship",
    "About the research project",
)


def main() -> int:
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                page.set_default_timeout(TIMEOUT_MS)
                response = page.goto(
                    PROJECT_URL, wait_until="load", timeout=TIMEOUT_MS
                )
                visible_text = page.locator("body").inner_text()
                searchable_text = " ".join(visible_text.split()).casefold()

                print(f"Final URL: {page.url}")
                print(f"Page title: {page.title()}")
                print(f"Visible text character count: {len(visible_text)}")
                for phrase in EXPECTED_PHRASES:
                    present = phrase.casefold() in searchable_text
                    print(f"{phrase}: {'present' if present else 'absent'}")

                if response is None:
                    print("No HTTP response received; stopping.", file=sys.stderr)
                    return 1
                print(f"HTTP status: {response.status}")
                if response.status >= 400:
                    print("HTTP failure; stopping without retry.", file=sys.stderr)
                    return 1
                return 0
            finally:
                browser.close()
    except TimeoutError as exc:
        print(f"Browser probe timed out; no retry: {exc}", file=sys.stderr)
        return 1
    except Error as exc:
        print(f"Browser probe failed; no retry: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
