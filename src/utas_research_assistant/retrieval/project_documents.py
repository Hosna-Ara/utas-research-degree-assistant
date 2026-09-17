"""One retrieval document per structured project; source records remain unchanged."""

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, HttpUrl

from utas_research_assistant.models import ProjectRecord

FundingStatus = Literal["funded", "no_stipend", "unknown"]
MONEY = re.compile(
    r"(?:[$£€]\s*\d[\d,.]*|\b(?:AUD|USD|GBP|EUR)\s*\d[\d,.]*|"
    r"\b\d[\d,.]*\s*(?:AUD|USD|GBP|EUR|dollars?)\b)", re.IGNORECASE
)


class ProjectDocument(BaseModel):
    document_id: str
    project_id: str
    text: str
    title: str
    description: str | None
    source_url: HttpUrl
    status: str | None
    degree_types: list[str]
    student_types: list[str]
    location: str | None
    scholarship_text: str | None
    funding_status: FundingStatus
    closing_date: str | None
    primary_supervisor: str | None
    research_categories: list[str]


def funding_status(scholarship_text: str | None) -> FundingStatus:
    """Retrieval heuristic, not a funding guarantee; explicit 'No stipend' wins."""
    text = scholarship_text or ""
    if re.search(r"\bno\s+stipend\b", text, re.IGNORECASE):
        return "no_stipend"
    if MONEY.search(text):
        return "funded"
    return "unknown"


def project_document(project: ProjectRecord) -> ProjectDocument:
    fields = {
        "title": "Title", "description": "Description",
        "research_categories": "Research categories", "degree_types": "Degree types",
        "student_types": "Student types", "location": "Location",
        "primary_supervisor": "Primary supervisor", "scholarship_text": "Scholarship information",
        "closing_date": "Closing date", "status": "Status",
    }
    metadata = project.model_dump(include=set(fields) | {"project_id", "source_url"})
    lines = []
    for field, label in fields.items():
        value = metadata[field]
        if value:
            rendered = "; ".join(value) if isinstance(value, list) else value
            lines.append(f"{label}: {rendered}")
    identity = f"id:{project.project_id}" if project.project_id else f"url:{project.source_url}"
    return ProjectDocument(
        **metadata, document_id="project-" + hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        text="\n\n".join(lines), funding_status=funding_status(project.scholarship_text),
    )


def build_project_documents(projects: list[ProjectRecord]) -> list[ProjectDocument]:
    documents = [project_document(project) for project in projects]
    if len({d.document_id for d in documents}) != len(documents):
        raise ValueError("Duplicate project identities; expected one document per project")
    return documents
