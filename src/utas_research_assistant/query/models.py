"""Validated structure produced by query understanding."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

GraphOperation = Literal[
    "project_by_id", "projects_by_supervisor", "multi_constraint_projects",
    "supervisors_with_multiple_projects", "supervisors_with_funded_ict_international_projects",
    "open_projects_by_category", "categories_with_both_degree_types",
    "supervisors_across_multiple_categories", "count_open_funded_international_ict_projects",
    "supervisor_by_name", "supervisor_research_fields", "supervisors_by_research_field",
    "ict_supervisors", "ict_supervisors_by_research_field", "supervisor_school", "supervisors_with_orcid", "projects_by_supervisor_research_field",
]


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    scope: Literal["general", "projects", "all", "supervisors"]
    search_query: str = Field(min_length=1)
    degree_type: str | None = None
    student_type: str | None = None
    location: str | None = None
    funding_status: str | None = None
    status: str | None = None
    research_category: str | None = None
    supervisor: str | None = None
    project_id: str | None = None
    intent: str = Field(min_length=1, max_length=80)
    confidence: float | None = Field(default=None, ge=0, le=1)
    planner_method: Literal["llm", "fallback"]

    @field_validator("search_query", "degree_type", "student_type", "location", "funding_status",
                     "status", "research_category", "supervisor", "project_id", "intent")
    @classmethod
    def reject_blank_strings(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("must not be blank")
        return value

    def project_filters(self) -> dict[str, str]:
        fields = ("degree_type", "student_type", "location", "funding_status", "status",
                  "research_category", "supervisor", "project_id")
        return {field: getattr(self, field) for field in fields if getattr(self, field) is not None}


class ReasoningPlan(BaseModel):
    """A validated retrieval/graph strategy; graph operations are an explicit allowlist."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    method: Literal["retrieval", "graph", "hybrid_graph"]
    scope: Literal["general", "projects", "all", "supervisors"]
    search_query: str = Field(min_length=1)
    degree_type: str | None = None
    student_type: str | None = None
    location: str | None = None
    funding_status: str | None = None
    status: str | None = None
    research_category: str | None = None
    supervisor: str | None = None
    project_id: str | None = None
    graph_operation: GraphOperation | None = None
    intent: str = Field(min_length=1, max_length=80)
    confidence: float | None = Field(default=None, ge=0, le=1)
    planner_method: Literal["llm", "fallback"]

    @field_validator("search_query", "degree_type", "student_type", "location", "funding_status",
                     "status", "research_category", "supervisor", "project_id", "intent")
    @classmethod
    def reject_blank_strings(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("must not be blank")
        return value

    def project_filters(self) -> dict[str, str]:
        fields = ("degree_type", "student_type", "location", "funding_status", "status",
                  "research_category", "supervisor", "project_id")
        return {field: getattr(self, field) for field in fields if getattr(self, field) is not None}
