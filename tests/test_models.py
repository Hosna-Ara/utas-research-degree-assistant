from datetime import datetime, timezone

import pytest

from utas_research_assistant.models import ProjectRecord


@pytest.mark.parametrize("scholarship_text", ["$34,315 pa", "Up to $60,000 pa"])
def test_project_record_can_be_instantiated(scholarship_text):
    fetched_at = datetime.now(timezone.utc)
    record = ProjectRecord(
        project_id="12167",
        title="Example research project",
        source_url="https://www.utas.edu.au/research/degrees/available-projects?id=12167",
        degree_types=["PhD"],
        scholarship_text=scholarship_text,
        fetched_at=fetched_at,
    )

    assert record.project_id == "12167"
    assert record.title == "Example research project"
    assert str(record.source_url).endswith("?id=12167")
    assert record.degree_types == ["PhD"]
    assert record.scholarship_text == scholarship_text
    assert record.fetched_at == fetched_at
    assert record.closing_date is None
    assert record.student_types == []
