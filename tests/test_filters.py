from datetime import datetime, timezone

import pytest

from utas_research_assistant.models import ProjectRecord
from utas_research_assistant.retrieval.project_documents import project_document
from utas_research_assistant.retrieval.filters import filter_projects


@pytest.fixture
def projects():
    return [project_document(ProjectRecord(
        project_id=str(i), title=f"Project {i}", source_url=f"https://www.utas.edu.au/?id={i}",
        fetched_at=datetime.now(timezone.utc), degree_types=["PhD", "Master by Research"] if i == 1 else ["PhD"],
        student_types=["International"] if i == 1 else ["Domestic"],
        location="Hobart; Launceston" if i == 1 else "Sydney",
        scholarship_text="$34,315 pa" if i == 1 else "No stipend",
        status="Applications open" if i == 1 else "Under assessment",
        research_categories=["Information and Communication Technology"] if i == 1 else ["Biology"],
        primary_supervisor="Doctor Smith" if i == 1 else "Doctor Jones",
    )) for i in (1, 2)]


@pytest.mark.parametrize("field,value", [
    ("degree_type", "Master by Research"), ("degree_type", "MRes"), ("degree_type", "Masters by Research"),
    ("student_type", "International"), ("student_type", "international student"),
    ("location", "HOBART"), ("location", "Launceston"),
    ("funding_status", "funded"), ("funding_status", "scholarship"),
    ("status", "open"), ("status", "APPLICATIONS OPEN"),
    ("research_category", "ICT"), ("research_category", "AI/ICT"),
    ("supervisor", "doctor smith"), ("project_id", "1"),
])
def test_individual_filters_and_aliases(projects, field, value):
    assert [p.project_id for p in filter_projects(projects, **{field: value})] == ["1"]


def test_composition_zero_results_and_no_stipend(projects):
    assert filter_projects(projects, degree_type="PhD", student_type="International", location="Hobart",
                           funding_status="funded", status="open") == [projects[0]]
    assert filter_projects(projects, degree_type="MRes", location="Sydney") == []
    assert filter_projects(projects, funding_status="No stipend") == [projects[1]]
    assert filter_projects(projects, supervisor="Smith") == []  # No substring matches.
    assert filter_projects(projects) == projects
    assert filter_projects([]) == []
    with pytest.raises(ValueError, match="Unknown"):
        filter_projects(projects, typo="x")
    with pytest.raises(ValueError, match="blank"):
        filter_projects(projects, location=" ")
