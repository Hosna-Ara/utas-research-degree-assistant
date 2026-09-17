import json

import requests
from pydantic import ValidationError
import pytest

from utas_research_assistant.generation.answer_generator import AnswerGenerator, graph_tool_summary
from utas_research_assistant.generation.answer_generator import _deterministic_count_answer, _safe_evidence_answer
from utas_research_assistant.generation.citations import build_citation_map, validate_citations
from utas_research_assistant.generation.models import AnswerResponse
from utas_research_assistant.generation.prompts import ANSWER_SYSTEM_PROMPT


class Provider:
    model = "qwen3:1.7b-test"

    def __init__(self, response=None, error=None):
        self.response, self.error = response, error
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        if self.error:
            raise self.error
        return self.response


def sample_evidence(**updates):
    evidence = {
        "reasoning_method": "retrieval", "planner_method": "llm", "graph_operation_used": None,
        "ranked_retrieval_evidence": [{"item_type": "general_chunk", "title": "Entry requirements",
                                       "text": "English requirement information.",
                                       "source_url": "https://www.utas.edu.au/example", "category": "entry_requirements",
                                       "provenance": [{"document_id": "doc-1"}]}],
        "graph_result": None, "source_urls": ["https://www.utas.edu.au/example"], "project_ids": [],
    }
    evidence.update(updates)
    return evidence


def test_citation_map_creation_and_source_metadata():
    mapping, sources = build_citation_map(sample_evidence())
    assert list(mapping) == ["S1"]
    assert sources[0]["source_url"] == "https://www.utas.edu.au/example"
    assert mapping["S1"]["provenance"] == [{"document_id": "doc-1"}]


def test_valid_citations_enforced_and_invalid_removed():
    cleaned, used = validate_citations("Supported [S1], fabricated [S9].", {"S1": {}})
    assert cleaned == "Supported [S1], fabricated ."
    assert used == ["S1"]


def test_answer_generator_preserves_sources_and_discards_unknown_citation():
    provider = Provider(json.dumps({"answer": "English information is available [S1] [S99].",
                                    "insufficient_evidence": False}))
    response = AnswerGenerator(provider).generate("What English information is available?", sample_evidence())
    assert response.answer == "English information is available [S1] ."
    assert response.citations == ["S1"]
    assert response.sources[0]["source_url"].endswith("example")
    assert provider.prompt["evidence"]["retrieval_evidence"][0]["citation_id"] == "S1"


def test_graph_tool_result_is_preserved_separately():
    evidence = sample_evidence(reasoning_method="graph", graph_operation_used="count_open_funded_international_ict_projects",
                               graph_result={"count": 17, "projects": []})
    result = AnswerGenerator(Provider(json.dumps({"answer": "17 projects [S1].", "insufficient_evidence": False})))
    response = result.generate("Count?", evidence)
    assert response.tool_used == "SPARQL Knowledge Graph"
    assert "17 matching projects" in response.tool_result
    assert result.last_generation_method == "deterministic_evidence_render"
    assert graph_tool_summary(evidence) == response.tool_result


def test_exact_graph_count_is_preserved_in_answer():
    evidence = sample_evidence(reasoning_method="graph", graph_operation_used="count_open_funded_international_ict_projects",
                               graph_result={"count": 17, "projects": []})
    mapping, _ = build_citation_map(evidence)
    assert _deterministic_count_answer(evidence, mapping) == "There are 17 matching open, funded international PhD projects in ICT [S1]."


def test_hybrid_graph_answer_uses_ranked_candidates_not_all_graph_matches():
    ranked = [{"item_type": "research_project", "title": f"Candidate {i}", "text": f"ICT PhD funded {i}",
               "source_url": f"https://example.org/{i}", "project_id": str(12000 + i)} for i in range(1, 7)]
    evidence = sample_evidence(reasoning_method="hybrid_graph", graph_operation_used="multi_constraint_projects",
                               graph_result=[{"project_id": str(12000 + i), "title": f"Candidate {i}"} for i in range(1, 20)],
                               ranked_retrieval_evidence=ranked, project_ids=[str(12000 + i) for i in range(1, 7)])
    response = AnswerGenerator(Provider(json.dumps({"answer": "Unrelated summary.", "insufficient_evidence": False}))).generate(
        "Find ICT PhD projects", evidence)
    assert "Candidate 1" in response.answer and "Candidate 5" in response.answer
    assert "Candidate 6" not in response.answer
    assert len(response.sources) == 5


def test_project_retrieval_answer_uses_ranked_project_evidence():
    ranked = [{"item_type": "research_project", "title": f"AI Candidate {i}", "text": f"AI PhD project {i}",
               "source_url": f"https://example.org/{i}", "project_id": str(12000 + i)} for i in range(1, 7)]
    evidence = sample_evidence(scope="projects", reasoning_method="retrieval", ranked_retrieval_evidence=ranked,
                               project_ids=[str(12000 + i) for i in range(1, 7)])
    response = AnswerGenerator(Provider(json.dumps({"answer": "All projects are funded and in Hobart.",
                                                    "insufficient_evidence": False}))).generate("Find AI PhD projects", evidence)
    assert "AI Candidate 1" in response.answer and "AI Candidate 5" in response.answer
    assert "Candidate 6" not in response.answer
    assert "funded" not in response.answer
    assert response.sources and all(source["source_url"] for source in response.sources)


def test_extractive_fallback_preserves_english_score_values():
    text = "English Language Requirements\nProof of English ability.\nIELTS (academic)\nMinimum Overall Score\n7.0\nNo band less than 6.5\nTOEFL\n94\nOR\nOther pathway"
    evidence = sample_evidence(ranked_retrieval_evidence=[{"item_type": "general_chunk", "title": "Entry requirements",
        "text": text, "source_url": "https://example.org/requirements"}])
    citations, _ = build_citation_map(evidence)
    answer, used = _safe_evidence_answer("What English score do I need for a PhD?", evidence, citations)
    assert "7.0" in answer and "94" in answer and "6.5" in answer
    assert used == ["S1"]


def test_extractive_fallback_returns_application_document_list():
    text = "At a minimum you will need -\nA copy of your passport\nA signed copy of the Supervisory support form\nA Research Proposal\nEvidence of your qualifications\nA Curriculum Vitae (CV)\nOnce you have prepared your documents, submit online."
    evidence = sample_evidence(ranked_retrieval_evidence=[{"item_type": "general_chunk", "title": "Research degrees",
        "text": text, "source_url": "https://example.org/apply"}])
    citations, _ = build_citation_map(evidence)
    answer, used = _safe_evidence_answer("What documents do I need when applying?", evidence, citations)
    assert "copy of your passport" in answer
    assert "Research Proposal" in answer
    assert used == ["S1"]


def test_extractive_fallback_keeps_scholarship_values_verbatim():
    text = "A top-up scholarship provides $5,000pa for 1 year. China Scholarship Council applicants apply by 11 September 2026."
    evidence = sample_evidence(ranked_retrieval_evidence=[{"item_type": "general_chunk", "title": "Scholarships",
        "text": text, "source_url": "https://example.org/scholarships"}])
    citations, _ = build_citation_map(evidence)
    answer, used = _safe_evidence_answer("What scholarships are available?", evidence, citations)
    assert "$5,000pa" in answer
    assert "11 September 2026" in answer
    assert used == ["S1"]


def test_insufficient_evidence_response_and_project_id_grounding():
    evidence = sample_evidence(reasoning_method="graph", graph_result={"project_id": "12259", "title": "Example",
                                   "source_url": "https://www.utas.edu.au/p?id=12259"}, project_ids=["12259"])
    response = AnswerGenerator(Provider(json.dumps({"answer": "Project 12259 is listed [S1]; project 99999 is not.",
                                                    "insufficient_evidence": True}))).generate("Q", evidence)
    assert "project 99999" not in response.answer.lower()
    assert response.project_ids == ["12259"]
    assert response.insufficient_evidence


def test_unavailable_ollama_returns_explicit_insufficient_answer():
    generator = AnswerGenerator(Provider(error=requests.ConnectionError("offline")))
    response = generator.generate("Q", sample_evidence())
    assert response.insufficient_evidence
    assert "does not contain enough information" in response.answer
    assert "offline" in response_generator_error(response)
    assert generator.last_generation_method == "insufficient_evidence_fallback"


def response_generator_error(response):
    # Public response remains user-facing; provider failure is captured on the generator separately.
    generator = AnswerGenerator(Provider(error=requests.ConnectionError("offline")))
    generator.generate("Q", sample_evidence())
    return generator.last_error


def test_malformed_response_falls_back_without_claims():
    generator = AnswerGenerator(Provider("not json"))
    response = generator.generate("Q", sample_evidence())
    assert response.insufficient_evidence
    assert response.citations == []
    assert generator.last_error


@pytest.mark.parametrize("question,text,expected", [
    ("What documents do I need to submit with my research degree application?",
     "At a minimum you will need -\nA copy of your passport\nA Research Proposal\nUse the Online Application System to submit your application.",
     "A copy of your passport"),
    ("How do I apply for a research degree at UTAS?",
     "How to apply\n\nEnsure you allow plenty of time to submit an application ahead of the closing date and check you meet all project specific criteria.",
     "allow plenty of time"),
])
def test_malformed_answer_uses_relevant_evidence_fallback(question, text, expected):
    evidence = sample_evidence(ranked_retrieval_evidence=[{
        "item_type": "general_chunk", "title": "Research degrees", "text": text,
        "source_url": "https://www.utas.edu.au/research/degrees",
    }])
    generator = AnswerGenerator(Provider("not valid JSON"))
    response = generator.generate(question, evidence)
    assert not response.insufficient_evidence
    assert expected in response.answer
    assert response.citations and response.citations[0] == response.sources[0]["citation_id"]
    assert response.sources[0]["source_url"] == "https://www.utas.edu.au/research/degrees"
    assert generator.last_generation_method == "extractive_or_ranked_evidence_fallback"


def test_malformed_answer_with_no_evidence_remains_insufficient():
    evidence = sample_evidence(ranked_retrieval_evidence=[], source_urls=[])
    response = AnswerGenerator(Provider("not valid JSON")).generate("What documents are needed?", evidence)
    assert response.insufficient_evidence
    assert response.citations == []


@pytest.mark.parametrize("question,project_fields,expected_phrase", [
    ("What are the complete project-specific eligibility and selection criteria for project 12259?",
     {"project_id": "12259", "title": "Phishing Detection", "source_url": "https://example.org/p/12259",
      "eligibility": None, "selection_criteria": None}, "does not include"),
    ("What are the detailed project application requirements for project 12259?",
     {"project_id": "12259", "title": "Phishing Detection", "source_url": "https://example.org/p/12259",
      "eligibility": None, "selection_criteria": None}, "does not include"),
    ("What are the recent publications and research impact of the supervisor of project 12259?",
     {"project_id": "12259", "title": "Phishing Detection", "source_url": "https://example.org/p/12259",
      "primary_supervisor": "Doctor Example"}, "publication histories"),
])
def test_capability_guard_safely_blocks_missing_project_and_supervisor_details(question, project_fields, expected_phrase):
    evidence = sample_evidence(reasoning_method="graph", graph_result=project_fields, project_ids=["12259"])
    provider = Provider(json.dumps({"answer": "unsupported claim", "insufficient_evidence": False}))
    response = AnswerGenerator(provider).generate(question, evidence)
    assert response.insufficient_evidence
    assert expected_phrase in response.answer
    assert "https://example.org/p/12259" == response.sources[0]["source_url"]
    assert response.citations and response.citations[0] == response.sources[0]["citation_id"]
    assert provider.prompt is None


def test_capability_guard_reports_dated_snapshot_for_live_change_question(tmp_path):
    from utas_research_assistant.generation.evidence_guard import assess_evidence_capability

    manifest = tmp_path / "snapshot_manifest.json"
    manifest.write_text(json.dumps({"snapshot_date": "2026-09-14"}), encoding="utf-8")
    gap = assess_evidence_capability("What projects were added to the UTAS website today?", {}, manifest)
    assert gap is not None and gap.capability == "live_website_changes"
    assert "2026-09-14" in gap.message
    assert "cannot verify live" in gap.message


def test_answer_response_validation():
    response = AnswerResponse(question="Q", answer="A", reasoning_method="retrieval", planner_method="llm",
                              citations=[], sources=[], insufficient_evidence=False, project_ids=[],
                              generation_model="qwen3:1.7b")
    assert response.generation_model == "qwen3:1.7b"
    with pytest.raises(ValidationError):
        AnswerResponse(question="Q", answer="A", reasoning_method="retrieval", planner_method="llm",
                       citations=[], sources=[], insufficient_evidence="no", project_ids=[],
                       generation_model="qwen3:1.7b")


def test_prompt_contains_grounding_guards():
    for phrase in ("ONLY the supplied structured evidence", "Never invent", "eligibility",
                   "guaranteed suitability", "Never fabricate citations"):
        assert phrase in ANSWER_SYSTEM_PROMPT
