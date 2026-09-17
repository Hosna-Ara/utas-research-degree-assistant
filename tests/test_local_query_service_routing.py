from pathlib import Path

import pytest

from utas_research_assistant.generation.evidence_guard import assess_evidence_capability
from utas_research_assistant.service import create_service
from utas_research_assistant.query.planner import fallback_reasoning_plan


@pytest.fixture(scope="module")
def service():
    return create_service(Path(__file__).resolve().parents[1] / "data" / "processed")


def _local_ids(result):
    return [
        row.get("document_id")
        for row in result.evidence.get("ranked_retrieval_evidence", [])
        if row.get("item_type") == "local_document"
    ]


def test_local_context_preserves_original_question():
    question = "What type of applicant is Applicant B?"
    plan = fallback_reasoning_plan(question)
    assert plan.intent == "local_document_lookup"
    assert plan.scope == "all"
    assert plan.search_query == question


def test_applicant_b_service_retrieval(service):
    result = service.answer_with_evidence("What type of applicant is Applicant B?")
    assert "LOCAL-BUSINESS-PERSONA-02" in _local_ids(result)
    assert _local_ids(result)[0] == "LOCAL-BUSINESS-PERSONA-02"


@pytest.mark.parametrize("question, expected_id", [
    ("Using my private applicant profile, summarise my background and main research interests.", "LOCAL-HOSNA-PROFILE-01"),
    ("What does my private applicant profile say about my research background?", "LOCAL-HOSNA-PROFILE-01"),
    ("Summarise my private profile.", "LOCAL-HOSNA-PROFILE-01"),
    ("Summarise the private applicant profile.", "LOCAL-HOSNA-PROFILE-01"),
    ("What type of applicant is Applicant B?", "LOCAL-BUSINESS-PERSONA-02"),
    ("Summarise the business-focused synthetic persona.", "LOCAL-BUSINESS-PERSONA-02"),
    ("What does my private project decision journal say?", "LOCAL-DECISION-JOURNAL-07"),
])
def test_exact_local_document_is_primary_evidence(service, question, expected_id):
    result = service.answer_with_evidence(question)
    primary = result.evidence["ranked_retrieval_evidence"][0]
    assert primary["item_type"] == "local_document"
    assert primary["document_id"] == expected_id
    assert primary["provenance"][0]["document_id"] == expected_id


def test_public_question_remains_public_retrieval(service):
    result = service.answer_with_evidence("What English score do I need for a PhD?")
    assert result.evidence.get("scope") == "general"
    assert not _local_ids(result)


def test_public_evidence_guard_does_not_expose_private_documents():
    gap = assess_evidence_capability(
        "What does my private applicant profile say about my research background?",
        {"ranked_retrieval_evidence": [], "project_ids": []},
    )
    assert gap is not None
    assert "active knowledge base" in gap.message
    assert "LOCAL-HOSNA-PROFILE-01" not in gap.message

