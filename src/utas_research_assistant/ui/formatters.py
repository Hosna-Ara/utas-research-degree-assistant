"""Small pure formatters that keep UI rendering separate from backend logic."""

from urllib.parse import urlparse


def official_source_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"www.utas.edu.au", "utas.edu.au"}:
        return None
    return url


def official_supervisor_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.hostname == "discover.utas.edu.au":
        return url
    return None


def project_metadata_from_source(source: dict) -> dict:
    """Flatten the matching provenance record while retaining citation fields."""
    metadata = dict(source)
    project_id = str(source.get("project_id") or "")
    for row in source.get("provenance", []) or []:
        if isinstance(row, dict) and str(row.get("project_id") or "") == project_id:
            metadata.update(row)
            break
    return metadata


def project_cards_from_sources(sources: list[dict]) -> list[dict]:
    cards = []
    seen = set()
    for source in sources:
        if not source.get("project_id"):
            continue
        project_id = str(source["project_id"])
        if project_id in seen:
            continue
        seen.add(project_id)
        card = project_metadata_from_source(source)
        card["project_id"] = project_id
        card["title"] = card.get("title") or source.get("title")
        card["source_url"] = official_source_url(card.get("source_url") or source.get("source_url"))
        card["supervisor_profile"] = source.get("supervisor_profile") or card.get("supervisor_profile")
        cards.append(card)
    return cards


def source_display_title(source: dict) -> str:
    if source.get("project_id"):
        title = source.get("title") or "Research project"
        return f"Project {source['project_id']} — {title}"
    return str(source.get("title") or "UTAS source")


def metadata_value(value) -> str | None:
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, (list, tuple, set)):
        value = ", ".join(str(item) for item in value if item not in (None, ""))
    rendered = str(value).strip()
    return rendered or None
