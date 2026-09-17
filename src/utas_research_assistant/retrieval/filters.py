"""Explicit, conjunctive project metadata filters; no query interpretation."""

from utas_research_assistant.retrieval.project_documents import ProjectDocument

FIELDS = {
    "degree_type": "degree_types", "student_type": "student_types", "location": "location",
    "funding_status": "funding_status", "status": "status", "research_category": "research_categories",
    "supervisor": "primary_supervisor", "project_id": "project_id",
}
ALIASES = {
    "degree_type": {"mres": "master by research", "masters by research": "master by research",
                    "ph.d.": "phd", "doctor of philosophy": "phd"},
    "student_type": {"international student": "international", "international students": "international",
                     "domestic student": "domestic", "domestic students": "domestic"},
    "funding_status": {"scholarship": "funded", "no stipend": "no_stipend"},
    "status": {"open": "applications open", "closed": "under assessment"},
    "research_category": {"ict": "information and communication technology",
                          "ai/ict": "information and communication technology",
                          "ai/ict-related": "information and communication technology"},
}


def canonical(field: str, value: str) -> str:
    value = " ".join(value.casefold().split())
    return ALIASES.get(field, {}).get(value, value)


def filter_projects(projects: list[ProjectDocument], **filters: str | None) -> list[ProjectDocument]:
    unknown = filters.keys() - FIELDS.keys()
    if unknown:
        raise ValueError(f"Unknown filters: {', '.join(sorted(unknown))}")
    constraints = {key: canonical(key, value) for key, value in filters.items() if value is not None}
    if any(not value for value in constraints.values()):
        raise ValueError("Filter values must not be blank")
    result = []
    for project in projects:
        for field, expected in constraints.items():
            value = getattr(project, FIELDS[field])
            values = value if isinstance(value, list) else (value or "").split(";") if field == "location" else [value or ""]
            if expected not in {canonical(field, entry) for entry in values}:
                break
        else:
            result.append(project)
    return result
