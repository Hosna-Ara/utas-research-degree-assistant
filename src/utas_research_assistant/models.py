"""Records for source information, preserving text until parsing is justified."""

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class ProjectRecord(BaseModel):
    """A project and its provenance; missing source details remain optional."""

    project_id: str
    title: str
    source_url: HttpUrl
    fetched_at: datetime
    status: str | None = None
    degree_types: list[str] = Field(default_factory=list)
    student_types: list[str] = Field(default_factory=list)
    location: str | None = None
    scholarship_text: str | None = None
    # Keep source wording, including values such as "Open until filled".
    closing_date: str | None = None
    primary_supervisor: str | None = None
    research_categories: list[str] = Field(default_factory=list)
    description: str | None = None
    eligibility: str | None = None
    selection_criteria: str | None = None
