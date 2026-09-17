"""Fetch a single page without crawling or automatic retries."""

import requests

USER_AGENT = (
    "UTASResearchDegreeAssistant/0.1 "
    "(unofficial student-built educational prototype)"
)


class FetchError(RuntimeError):
    """A source page could not be retrieved successfully."""


def fetch_page(
    url: str,
    *,
    session: requests.Session,
    timeout: float = 30.0,
) -> requests.Response:
    """Fetch one URL using a caller-owned session and a timeout in seconds."""
    try:
        response = session.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=timeout
        )
        response.raise_for_status()
        return response
    except requests.Timeout as exc:
        raise FetchError(f"Timed out fetching {url} after {timeout}s") from exc
    except requests.HTTPError as exc:
        raise FetchError(f"HTTP error fetching {url}: {exc}") from exc
    except requests.RequestException as exc:
        raise FetchError(f"Request failed for {url}: {exc}") from exc
