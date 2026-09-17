"""Run with: python scripts/probe_utas.py (after installing requirements)."""

from pathlib import Path
import sys

import requests
from bs4 import BeautifulSoup

# Allow direct execution from this source checkout without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utas_research_assistant.config import PROJECTS_URL
from utas_research_assistant.ingestion.fetch import FetchError, fetch_page

EXPECTED_PHRASES = (
    "Degree type",
    "Student type",
    "Scholarship",
    "About the research project",
)


def probe_page(url: str, session: requests.Session) -> bool:
    """Report basic server-rendered page checks without extracting records."""
    print(f"URL: {url}")
    try:
        response = fetch_page(url, session=session)
    except FetchError as exc:
        print(f"HTTP success: no — {exc}")
        return False

    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    page_text = " ".join(soup.get_text(" ", strip=True).split()).casefold()

    print(f"HTTP success: yes ({response.status_code})")
    print(f"Page title: {soup.title.get_text(strip=True) if soup.title else '(missing)'}")
    print(f"Downloaded character count: {len(html)}")
    for phrase in EXPECTED_PHRASES:
        print(f"{phrase}: {'present' if phrase.casefold() in page_text else 'absent'}")
    return True


def main() -> int:
    with requests.Session() as session:
        projects_ok = probe_page(PROJECTS_URL, session)
        print()
        example_ok = probe_page(f"{PROJECTS_URL}?id=12167", session)
    return 0 if projects_ok and example_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
