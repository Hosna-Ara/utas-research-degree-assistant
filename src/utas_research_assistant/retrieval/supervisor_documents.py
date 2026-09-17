"""Validation model for normalized supervisor retrieval documents."""

from pydantic import BaseModel, Field, HttpUrl


class SupervisorDocument(BaseModel):
    document_id: str
    supervisor_id: str
    text: str
    title: str
    source_url: HttpUrl
    discovery_profile_id: str | None = None
    related_project_ids: list[str] = Field(default_factory=list)
    related_research_categories: list[str] = Field(default_factory=list)
    is_ict_supervisor: bool = False
    canonical_name: str
    school: str | None = None
    research_fields: list[str] = Field(default_factory=list)
    orcid: str | None = None
    google_scholar_url: str | None = None
