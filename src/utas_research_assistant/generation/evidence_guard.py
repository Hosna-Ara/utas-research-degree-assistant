"""Detect requests that require evidence this local snapshot does not contain."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

from utas_research_assistant.retrieval.local_documents import has_local_document_reference


@dataclass(frozen=True)
class EvidenceGap:
    capability: str
    message: str
    project_id: str | None = None
    source_url: str | None = None
    snapshot_date: str | None = None


def _project_evidence(evidence: dict) -> list[dict]:
    rows = list(evidence.get("ranked_retrieval_evidence", []) or [])

    def visit(value):
        if isinstance(value, dict):
            if value.get("project_id"):
                rows.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(evidence.get("graph_result"))
    return rows


def _requested_project_id(question: str, evidence: dict) -> str | None:
    match = re.search(r"\bproject\s*(?:id\s*)?#?\s*(\d{4,8})\b", question, re.I)
    if match:
        return match.group(1)
    ids = list(map(str, evidence.get("project_ids", [])))
    return ids[0] if len(ids) == 1 else None


def _snapshot_date(manifest_path: Path | None = None) -> str | None:
    if manifest_path is None:
        root = Path(__file__).resolve().parents[3]
        manifest_path = root / "data" / "processed" / "snapshot_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("snapshot_date")
    except (OSError, ValueError, AttributeError):
        return None


def assess_evidence_capability(question: str, evidence: dict,
                               manifest_path: Path | None = None) -> EvidenceGap | None:
    """Return a safe capability response when requested evidence is unavailable."""
    text = " ".join(question.casefold().split())
    projects = _project_evidence(evidence)
    project_id = _requested_project_id(question, evidence)
    matching = [row for row in projects if project_id is None or str(row.get("project_id")) == project_id]
    source_url = next((row.get("source_url") for row in matching if row.get("source_url")), None)

    local_request = bool(re.search(
        r"\b(?:private|local|uploaded|user[- ]authored|assignment|persona|decision\s+journal)\b"
        r"|\bcandidate\b.{0,40}\b(?:rejected|declined)\b",
        text,
    ))
    has_local_evidence = any(row.get("item_type") == "local_document"
                             for row in evidence.get("ranked_retrieval_evidence", []) or [])
    if (local_request or has_local_document_reference(question)) and not has_local_evidence:
        return EvidenceGap(
            "local_document_unavailable",
            "The active knowledge base does not contain the requested private or local document information.",
        )

    detailed_criteria = bool(re.search(
        r"\b(eligib(?:ility|le)|selection\s+criteria|selection\s+process|"
        r"detailed\s+(?:project\s+)?(?:application\s+)?requirements?|"
        r"project\s+application\s+requirements?|complete\s+criteria)\b", text,
    ))
    project_specific = bool(project_id or re.search(r"\bproject[- ]specific\b|\bfor (?:this|the) project\b", text))
    if detailed_criteria and project_specific:
        has_detail = any(row.get("eligibility") or row.get("selection_criteria") for row in matching)
        if not has_detail:
            target = f" for project {project_id}" if project_id else " for this project"
            message = (f"The local UTAS project listing{target} contains summary information, but it does not include "
                       "the detailed eligibility or selection criteria requested. Check the official project listing for updates.")
            return EvidenceGap("project_detail_criteria", message, project_id, source_url)

    profile_request = bool(re.search(
        r"\b(publications?|publication history|research profile|research impact|citation metrics?|"
        r"h[- ]index|research outputs?)\b", text,
    ))
    supervisor_context = bool(re.search(r"\bsupervisor\w*\b|\bprofile\b", text))
    has_supervisor_evidence = any(row.get("item_type") == "supervisor_profile" for row in evidence.get("ranked_retrieval_evidence", []) or [])
    if profile_request and (supervisor_context or has_supervisor_evidence):
        message = ("Information is not available in the current knowledge snapshot. The assistant includes supervisor "
                   "profile information, but publication histories, h-index values, live citation metrics, and "
                   "complete publication records have not been ingested, so I cannot verify those details.")
        return EvidenceGap("supervisor_profile_publications", message, project_id, source_url)

    live_change_request = bool(
        re.search(r"\b(?:added|new|changes?|changed|updated|updates?)\b.{0,45}\b(?:today|latest|recent|"
                  r"website|site|since|snapshot|currently)\b", text)
        or re.search(r"\b(?:today|latest|recent)\b.{0,45}\b(?:projects?|website|site|changes?|updates?)\b", text)
        or re.search(r"\bwhat(?:'s| is) currently new since\b", text)
    )
    if live_change_request:
        date = _snapshot_date(manifest_path)
        date_note = f" The available project snapshot is dated {date}." if date else ""
        message = ("The local dated snapshot cannot verify live UTAS website changes or determine what has been added "
                   f"since it was saved.{date_note}")
        return EvidenceGap("live_website_changes", message, snapshot_date=date)
    return None
