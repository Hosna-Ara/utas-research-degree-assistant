"""Parse saved UTAS listing cards; never fetch pages or execute JavaScript."""

import json
import logging
import re
from datetime import datetime
from urllib.parse import parse_qs, urljoin, urlsplit

from bs4 import BeautifulSoup, Tag
from pydantic import ValidationError

from utas_research_assistant.config import BASE_URL, PROJECTS_URL
from utas_research_assistant.models import ProjectRecord

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Collapse whitespace without changing wording or punctuation."""
    return " ".join(text.split())


def node_text(node: Tag | None) -> str:
    return clean_text(node.get_text(" ", strip=True)) if node else ""


def project_link(href: str) -> tuple[str, str]:
    """Resolve UTAS links and return the query ID, or an empty ID if absent."""
    href = href.strip()
    if not href or href.startswith("#"):
        raise ValueError("Missing project URL")
    # Query-only links refer to the listing page, not the site root.
    url = urljoin(PROJECTS_URL if href.startswith("?") else BASE_URL, href)
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or parts.hostname != "www.utas.edu.au":
        raise ValueError(f"Not a public UTAS project URL: {url}")
    project_id = parse_qs(parts.query).get("id", [""])[0].strip()
    return project_id, url


def split_types(text: str) -> list[str]:
    """Split the listing's 'and' separator only; retain each type's wording."""
    return [part for part in re.split(r"\s+and\s+", clean_text(text)) if part]


def listing_locations(soup: BeautifulSoup) -> dict[str, str]:
    """Read location metadata from the saved hdrData JSON array, without eval."""
    locations = {}
    for script in soup.find_all("script"):
        text = script.get_text()
        match = re.search(r"\bconst\s+hdrData\s*=\s*", text)
        if not match:
            continue
        try:
            rows, _ = json.JSONDecoder().raw_decode(text[match.end():])
            if not isinstance(rows, list):
                raise ValueError("hdrData is not a list")
        except ValueError as exc:
            logger.warning("Cannot read embedded listing locations: %s", exc)
            continue
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                logger.warning("Invalid embedded listing row %d; no location used", index)
                continue
            values = row.get("locations", [])
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                logger.warning("Invalid locations for embedded project %r", row.get("id"))
                continue
            if row.get("id") is not None:
                locations[str(row["id"])] = "; ".join(
                    clean_text(value) for value in values if clean_text(value)
                )
    return locations


def parse_project_card(
    card: Tag, *, fetched_at: datetime, locations: dict[str, str] | None = None
) -> ProjectRecord:
    """Validate one listing card; unavailable values remain empty."""
    link = card.select_one("a.hdr-project__card--title")
    if link is None or not node_text(link):
        raise ValueError("Missing project title")
    project_id, source_url = project_link(str(link.get("href", "")))
    details = {}
    for item in card.select(".hdr-project__card--details--item"):
        label = item.select_one(".hdr-project__card--details--label")
        if label is None:
            logger.warning("Unlabelled detail in project %s", source_url)
            continue
        key = node_text(label).rstrip(":").casefold()
        # Omit the label node without mutating the original soup.
        value = clean_text(" ".join(
            str(text) for text in item.strings if label not in text.parents
        ))
        details[key] = value

    status = card.select_one(
        ".hdr-project__card--status .open, "
        ".hdr-project__card--status .closed, "
        ".hdr-project__card--status .specific"
    )
    return ProjectRecord(
        project_id=project_id,
        title=node_text(link),
        source_url=source_url,
        fetched_at=fetched_at,
        status=node_text(status) or None,
        degree_types=split_types(details.get("degree type", "")),
        student_types=split_types(details.get("student type", "")),
        location=details.get("location") or (locations or {}).get(project_id) or None,
        scholarship_text=details.get("scholarship") or None,
        closing_date=details.get("closing date") or None,
        primary_supervisor=details.get("primary supervisor") or None,
        research_categories=[
            node_text(category)
            for category in card.select(".project__card--category")
            if node_text(category)
        ],
        description=node_text(card.select_one(".hdr-project__card--text")) or None,
    )


def parse_projects(html: str, *, fetched_at: datetime) -> list[ProjectRecord]:
    """Parse all saved cards, retaining the first valid record per ID or URL."""
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select(".hdr-project__listing .hdr-project__card")
    if not cards:
        raise ValueError("No UTAS project listing cards found; check the snapshot structure")
    locations = listing_locations(soup)
    records = []
    seen = set()
    skipped = duplicates = 0
    for index, card in enumerate(cards, 1):
        try:
            record = parse_project_card(card, fetched_at=fetched_at, locations=locations)
        except ValidationError as exc:
            skipped += 1
            logger.warning("Card %d validation failed; skipped: %s", index, exc)
            continue
        except (ValueError, TypeError) as exc:
            skipped += 1
            logger.warning("Malformed card %d skipped: %s", index, exc)
            continue
        key = ("id", record.project_id) if record.project_id else ("url", str(record.source_url))
        if key in seen:
            duplicates += 1
            logger.warning("Duplicate card %d (%s); keeping first valid record", index, key[1])
            continue
        if not record.project_id:
            logger.warning("Card %d has no project ID; using URL for deduplication", index)
        seen.add(key)
        records.append(record)
    logger.info(
        "Listing cards: %d; parsed: %d; malformed/invalid: %d; duplicates: %d",
        len(cards), len(records), skipped, duplicates,
    )
    if skipped:
        logger.warning("Skipped %d of %d listing cards", skipped, len(cards))
    if not records:
        raise ValueError("All listing cards failed parsing or validation")
    return records
