from utas_research_assistant.graph.builder import build_graph, graph_statistics
from utas_research_assistant.graph.queries import (
    get_projects_by_supervisor, get_supervisor_profile,
    get_supervisor_research_fields, get_supervisors_by_school,
    get_supervisors_with_orcid,
)
from utas_research_assistant.retrieval.corpus import build_corpus
from utas_research_assistant.retrieval.hybrid import HybridRetriever
from utas_research_assistant.retrieval.supervisor_documents import SupervisorDocument
from utas_research_assistant.retrieval.project_documents import ProjectDocument
from utas_research_assistant.generation.answer_generator import AnswerGenerator
import requests


def project(project_id="12103"):
    return ProjectDocument(
        document_id=f"project-{project_id}", project_id=project_id,
        title="Example project", text="Example project about artificial intelligence",
        source_url=f"https://www.utas.edu.au/research/degrees/available-projects?id={project_id}",
        status="Applications open", degree_types=["PhD"], student_types=["International"],
        location="Hobart", scholarship_text="$34,315 pa", funding_status="funded",
        closing_date=None, primary_supervisor="Doctor Soonja Yeom",
        research_categories=["Information and Communication Technology"], description="Example",
    )


def supervisor():
    return SupervisorDocument(
        document_id="supervisor-soonja-yeom", supervisor_id="soonja-yeom",
        title="Doctor Soonja Yeom", canonical_name="Doctor Soonja Yeom",
        text="Doctor Soonja Yeom. Research fields: Cybersecurity and privacy; Education.",
        source_url="https://discover.utas.edu.au/api/users/Soonja.Yeom",
        discovery_profile_id="Soonja.Yeom", related_project_ids=["12103"],
        related_research_categories=["Information and Communication Technology"],
        school="School of Information and Communication Technology",
        research_fields=["Cybersecurity and privacy", "Education"], is_ict_supervisor=True,
    )


def test_supervisor_profile_is_indexed_with_metadata_and_scope():
    corpus = build_corpus([], [project().model_dump(mode="json")], [supervisor().model_dump(mode="json")])

    class NoSemantic:
        def search(self, query, top_k=5, *, candidate_indices=None):
            return []

    results = HybridRetriever(corpus, None, NoSemantic()).search("Soonja Yeom", scope="supervisors")
    assert results and results[0]["item_type"] == "supervisor_profile"
    assert results[0]["canonical_name"] == "Doctor Soonja Yeom"
    assert results[0]["related_project_ids"] == ["12103"]


def test_enriched_graph_reconciles_title_variants_and_exposes_profile_facts():
    graph = build_graph([project()], [supervisor()])
    assert graph_statistics(graph)["number_of_supervisors"] == 1
    assert len(get_projects_by_supervisor(graph, "Soonja Yeom")) == 1
    profile = get_supervisor_profile(graph, "Dr Soonja Yeom")
    assert profile["canonical_name"] == "Doctor Soonja Yeom"
    assert "Cybersecurity and privacy" in get_supervisor_research_fields(graph, "soonja yeom")["research_fields"]
    assert get_supervisors_by_school(graph, "School of Information and Communication Technology")


def test_supervisor_orcid_query_is_deterministic():
    profile = supervisor().model_dump(mode="json")
    profile["orcid"] = "0000-0001-2345-6789"
    graph = build_graph([project()], [profile])
    assert get_supervisors_with_orcid(graph) == [{"supervisor": "Doctor Soonja Yeom", "orcid": "0000-0001-2345-6789"}]


def test_unresolved_supervisor_profile_does_not_break_project_relationship():
    graph = build_graph([project()], [])
    assert len(get_projects_by_supervisor(graph, "Doctor Soonja Yeom")) == 1
    assert get_supervisor_profile(graph, "Soonja Yeom")["discovery_url"] is None


def test_supervisor_graph_answer_is_cited_and_profile_facts_are_grounded():
    class FakeProvider:
        model = "test"
        def generate(self, prompt):
            return '{"answer":"placeholder [S1]","insufficient_evidence":false}'

    evidence = {
        "reasoning_method": "graph", "planner_method": "fallback",
        "graph_operation_used": "supervisor_school", "scope": "supervisors",
        "graph_result": {"canonical_name": "Doctor Quan Bai",
                         "school": "School of Information and Communication Technology",
                         "source_url": "https://discover.utas.edu.au/api/users/Quan.Bai"},
        "ranked_retrieval_evidence": [], "project_ids": [], "source_urls": [],
    }
    response = AnswerGenerator(FakeProvider()).generate("Which school is Quan Bai affiliated with?", evidence)
    assert "School of Information and Communication Technology" in response.answer
    assert response.citations == ["S1"]
    assert not response.insufficient_evidence


def test_project_supervisor_graph_answer_survives_unavailable_answer_model():
    class UnavailableProvider:
        model = "test"
        def generate(self, prompt):
            raise requests.RequestException("offline")

    evidence = {
        "reasoning_method": "graph", "planner_method": "fallback",
        "graph_operation_used": "projects_by_supervisor", "scope": "projects",
        "graph_result": [{"project_id": "12103", "title": "Example project",
                          "source_url": "https://example.org/12103"}],
        "ranked_retrieval_evidence": [], "project_ids": ["12103"], "source_urls": [],
    }
    response = AnswerGenerator(UnavailableProvider()).generate("Which projects are supervised by Soonja Yeom?", evidence)
    assert "12103" in response.answer
    assert response.citations
    assert not response.insufficient_evidence
