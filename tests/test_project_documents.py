from datetime import datetime, timezone

import pytest

from utas_research_assistant.models import ProjectRecord
from utas_research_assistant.retrieval.project_documents import (
    build_project_documents, funding_status, project_document,
)


def project(**overrides):
    values = dict(
        project_id="12167", title="Marine research", description="Study coastal habitats.",
        source_url="https://www.utas.edu.au/research/degrees/available-projects?id=12167",
        fetched_at=datetime(2026, 9, 15, tzinfo=timezone.utc), status="Applications open",
        degree_types=["PhD", "Master by Research"], student_types=["Domestic", "International"],
        location="Hobart", primary_supervisor="Doctor Example", scholarship_text="Up to $60,000 pa",
        closing_date="1 October 2026", research_categories=["Marine and Antarctic"],
    )
    return ProjectRecord(**(values | overrides))


@pytest.mark.parametrize("value, expected", [
    ("$34,315 pa", "funded"), ("Up to $60,000 pa", "funded"),
    ("AUD 34315 per year", "funded"), ("34,315 AUD", "funded"),
    ("No stipend", "no_stipend"), (" no STIPEND ", "no_stipend"),
    ("No stipend; $500 travel support", "no_stipend"),
    (None, "unknown"), ("", "unknown"), ("Scholarship", "unknown"),
    ("N/A", "unknown"), ("Apply in 2026", "unknown"),
])
def test_funding_status(value, expected):
    assert funding_status(value) == expected


def test_project_text_metadata_and_source_unchanged():
    source = project()
    before = source.model_dump()
    record = project_document(source)
    for field in before.keys() - {"fetched_at", "eligibility", "selection_criteria"}:
        assert getattr(record, field) == getattr(source, field)
    assert record.funding_status == "funded"
    assert "Title: Marine research" in record.text
    assert "Description: Study coastal habitats." in record.text
    assert "Degree types: PhD; Master by Research" in record.text
    assert "Scholarship information: Up to $60,000 pa" in record.text
    for field in ("status", "location", "primary_supervisor", "closing_date"):
        assert getattr(source, field) in record.text
    assert "Research categories: Marine and Antarctic" in record.text
    assert "Student types: Domestic; International" in record.text
    assert source.model_dump() == before


def test_stable_identity_optional_metadata_and_duplicates():
    source = project(scholarship_text=None, description=None, primary_supervisor=None)
    record = project_document(source)
    assert record == project_document(source)
    assert record.scholarship_text is None
    assert record.description is None
    assert "Scholarship information:" not in record.text
    assert record.funding_status == "unknown"
    source.title = "Updated title"
    assert project_document(source).document_id == record.document_id
    fallback = project(project_id="")
    assert project_document(fallback).document_id == project_document(fallback).document_id
    assert project_document(project(project_id="another")).document_id != record.document_id
    assert len(build_project_documents([source, project(project_id="another")])) == 2
    with pytest.raises(ValueError, match="Duplicate project identities"):
        build_project_documents([source, source])
