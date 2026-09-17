"""Conservative UTAS Discover supervisor-profile acquisition and normalization."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, HttpUrl

DISCOVER_API = "https://discover.utas.edu.au/api/users"
DISCOVER_BASE = "https://discover.utas.edu.au"
USER_AGENT = "UTAS-Research-Degree-Assistant/0.1 (educational prototype)"
TITLE_PATTERN = re.compile(r"^(?:associate\s+professor|assoc\.?\s+prof|professor|prof\.?|doctor|dr\.?)\s+", re.I)


class SupervisorTarget(BaseModel):
    canonical_project_supervisor_name: str
    normalized_name: str
    project_ids: list[str] = Field(default_factory=list)
    research_categories: list[str] = Field(default_factory=list)
    is_ict_supervisor: bool = False


class SupervisorProfile(BaseModel):
    supervisor_id: str
    canonical_name: str
    first_name: str | None = None
    last_name: str | None = None
    title: str | None = None
    discovery_profile_id: str | None = None
    discovery_url: HttpUrl | None = None
    school: str | None = None
    sub_organisation: str | None = None
    positions: list[Any] = Field(default_factory=list)
    bio: str | None = None
    research_fields: list[str] = Field(default_factory=list)
    tags: list[Any] = Field(default_factory=list)
    orcid: str | None = None
    email: str | None = None
    degrees: list[Any] = Field(default_factory=list)
    academic_appointments: list[Any] = Field(default_factory=list)
    availability: Any = None
    linkedin_url: str | None = None
    google_scholar_url: str | None = None
    researchgate_url: str | None = None
    elements_profile_url: str | None = None
    grants_summary: str | None = None
    teaching_summary: str | None = None
    updated_when: str | None = None
    source_url: HttpUrl
    fetched_at: datetime
    related_project_ids: list[str] = Field(default_factory=list)
    related_research_categories: list[str] = Field(default_factory=list)
    is_ict_supervisor: bool = False
    resolution_status: str = "resolved"
    attempted_profile_ids: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


def strip_academic_titles(name: str) -> str:
    """Remove only leading, known academic honorifics."""
    value = " ".join(str(name or "").strip().split())
    while True:
        stripped = TITLE_PATTERN.sub("", value, count=1).strip()
        if stripped == value:
            return value
        value = stripped


def normalize_name(name: str) -> str:
    """Canonical comparison form; conservative and suitable for exact matching."""
    value = strip_academic_titles(name)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[’'`.-]+", " ", value)
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _display_token(token: str) -> str:
    return token[:1].upper() + token[1:]


def candidate_profile_ids(name: str) -> list[str]:
    """Generate a small deterministic set of plausible Discover identifiers."""
    clean = strip_academic_titles(name)
    tokens = [t for t in re.split(r"\s+", clean) if t]
    if len(tokens) < 2:
        return []
    shown = [_display_token(t) for t in tokens]
    candidates = [".".join(shown)]
    first, rest = shown[0], shown[1:]
    joined = "".join(rest)
    candidates.extend([f"{first}.{joined}", f"{first}.{joined[:1].upper()}{joined[1:]}"])
    # Discover occasionally publishes a compound given name under a shorter
    # surname-first identifier (for example, Bilal.Amin or Yang.Wenli).
    if len(shown) >= 3:
        candidates.extend([f"{shown[-2]}.{shown[-1]}", f"{shown[-1]}.{shown[0]}"])
    candidates.append(f"{shown[-1]}.{shown[0]}")
    # Preserve a hyphenated surname as a useful first candidate variant.
    candidates.append(f"{first}.{'-'.join(rest)}")
    return list(dict.fromkeys(candidates))


def profile_matches(profile: dict[str, Any], expected_name: str) -> bool:
    expected = normalize_name(expected_name)
    if not expected:
        return False
    values = [profile.get("firstNameLastName")]
    first, last = profile.get("firstName"), profile.get("lastName")
    if first or last:
        values.append(f"{first or ''} {last or ''}")
    return any(normalize_name(str(value)) == expected for value in values if value)


def clean_bio_html(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, dict):
        value = value.get("htmlStripped") or value.get("value") or value.get("text")
    if not value:
        return None
    soup = BeautifulSoup(str(value), "html.parser")
    for node in soup.select("script, style, noscript"):
        node.decompose()
    for node in soup.find_all(["p", "div", "li", "br"]):
        node.insert_before("\n")
    # Use an empty separator so inline emphasis does not become artificial
    # line breaks; explicit block markers above carry paragraph boundaries.
    text = "\n".join(" ".join(line.split()) for line in soup.get_text("").splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("value") or value.get("address") or value.get("uri") or value.get("name")
    result = " ".join(str(value).split()).strip()
    return result or None


def _source_values(value: Any) -> list[str]:
    """Flatten source-provided tag/label structures without deriving expertise."""
    values: list[str] = []
    if isinstance(value, dict):
        for key in ("explicit", "implicit", "value"):
            if key in value:
                values.extend(_source_values(value[key]))
    elif isinstance(value, list):
        for item in value:
            values.extend(_source_values(item))
    elif value is not None:
        text = _text(value)
        if text:
            values.append(text)
    return list(dict.fromkeys(values))


def _find_url(value: Any, terms: tuple[str, ...]) -> str | None:
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        low = value.casefold()
        if any(term in low for term in terms):
            return value
    if isinstance(value, dict):
        for child in value.values():
            found = _find_url(child, terms)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_url(child, terms)
            if found:
                return found
    return None


def _organisation_fields(addresses: Any) -> tuple[str | None, str | None]:
    school = sub = None
    for item in _as_list(addresses):
        if not isinstance(item, dict):
            continue
        values = {str(k).casefold(): v for k, v in item.items()}
        school = school or _text(values.get("school") or values.get("organisation") or values.get("organization"))
        sub = sub or _text(values.get("suborganisation") or values.get("sub-organisation") or values.get("department"))
    return school, sub


def normalize_profile(payload: dict[str, Any], target: SupervisorTarget, profile_id: str, fetched_at: datetime | None = None) -> SupervisorProfile:
    first = _text(payload.get("firstName"))
    last = _text(payload.get("lastName"))
    school, sub = _organisation_fields(payload.get("addresses"))
    source_url = f"{DISCOVER_API}/{quote(profile_id, safe='.') }"
    return SupervisorProfile(
        supervisor_id=f"supervisor-{target.normalized_name.replace(' ', '-')}",
        canonical_name=target.canonical_project_supervisor_name,
        first_name=first, last_name=last, title=_text(payload.get("title")),
        discovery_profile_id=profile_id, discovery_url=source_url, school=school,
        sub_organisation=sub, positions=_as_list(payload.get("positions")),
        bio=clean_bio_html(payload.get("tabSummaryAbout")),
        research_fields=_source_values(payload.get("tags") or payload.get("labels")),
        tags=_as_list(payload.get("tags")), orcid=_text(payload.get("orcid")),
        email=_text(payload.get("emailAddress")), degrees=_as_list(payload.get("degrees")),
        academic_appointments=_as_list(payload.get("academicAppointments")),
        availability=payload.get("availability"),
        linkedin_url=_find_url(payload.get("personalWebsites"), ("linkedin",)),
        google_scholar_url=_find_url(payload.get("personalWebsites"), ("scholar.google", "google scholar")),
        researchgate_url=_find_url(payload.get("personalWebsites"), ("researchgate",)),
        elements_profile_url=_text(payload.get("elementsUserProfileUrl")),
        grants_summary=clean_bio_html(payload.get("tabSummaryGrants")),
        teaching_summary=clean_bio_html(payload.get("tabSummaryTeachingActivities")),
        updated_when=_text(payload.get("updatedWhen")), source_url=source_url,
        fetched_at=fetched_at or datetime.now(timezone.utc), related_project_ids=target.project_ids,
        related_research_categories=target.research_categories, is_ict_supervisor=target.is_ict_supervisor,
    )


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def fetch_profile(session: requests.Session, profile_id: str, timeout: float = 20.0, max_retries: int = 2) -> tuple[int, dict[str, Any] | None, str | None]:
    """Fetch one profile, retrying only transient HTTP/server failures."""
    url = f"{DISCOVER_API}/{quote(profile_id, safe='.') }"
    for attempt in range(max_retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(1.0 + attempt)
                continue
            if response.status_code != 200:
                return response.status_code, None, f"HTTP {response.status_code}"
            try:
                payload = response.json()
            except ValueError:
                return response.status_code, None, "HTTP 200 response was not valid JSON"
            if not isinstance(payload, dict):
                return response.status_code, None, "HTTP 200 response was not a JSON object"
            return response.status_code, payload, None
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(1.0 + attempt)
                continue
            return 0, None, str(exc)
    return 0, None, "request failed"


def resolve_target(session: requests.Session, target: SupervisorTarget, timeout: float = 20.0, delay: float = 1.2) -> dict[str, Any]:
    attempts = []
    for candidate in candidate_profile_ids(target.canonical_project_supervisor_name):
        status, payload, error = fetch_profile(session, candidate, timeout=timeout)
        attempts.append({"profile_id": candidate, "url": f"{DISCOVER_API}/{quote(candidate, safe='.')}", "http_status": status, "reason": error})
        if payload is not None and profile_matches(payload, target.canonical_project_supervisor_name):
            return {"target": target, "profile_id": candidate, "payload": payload, "attempts": attempts}
        if delay:
            time.sleep(delay)
    return {"target": target, "profile_id": None, "payload": None, "attempts": attempts, "failure_reason": "No validated profile matched the supervisor name"}


def load_cached_profile(path: Path, target: SupervisorTarget) -> SupervisorProfile | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not profile_matches(payload, target.canonical_project_supervisor_name):
            return None
        profile_id = path.stem
        return normalize_profile(payload, target, profile_id, datetime.fromtimestamp(path.stat().st_mtime, timezone.utc))
    except (OSError, ValueError, TypeError):
        return None


def ict_quality_gate(targets: list[SupervisorTarget], profiles: list[SupervisorProfile]) -> bool:
    """Return true only when every ICT target has a validated profile."""
    ict_names = {target.normalized_name for target in targets if target.is_ict_supervisor}
    resolved_names = {normalize_name(profile.canonical_name) for profile in profiles if profile.is_ict_supervisor and profile.resolution_status == "resolved"}
    return ict_names <= resolved_names
