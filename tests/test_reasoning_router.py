import json

import pytest
from pydantic import ValidationError
from rdflib import Graph

from utas_research_assistant.graph.builder import build_graph
from utas_research_assistant.query.models import ReasoningPlan
from utas_research_assistant.query.planner import ReasoningPlanner, fallback_reasoning_plan
from utas_research_assistant.query.reasoning_router import ReasoningRouter, execute_graph_operation
from utas_research_assistant.retrieval.project_documents import ProjectDocument


def project(project_id="101", **changes):
    values = {
        "document_id": f"project-{project_id}", "project_id": project_id,
        "text": "AI machine learning project", "title": f"AI Project {project_id}",
        "description": "Research description", "source_url": f"https://example.org/{project_id}",
        "status": "Applications open", "degree_types": ["PhD"],
        "student_types": ["International"], "location": "Hobart",
        "scholarship_text": "$34,315 pa", "funding_status": "funded",
        "closing_date": None, "primary_supervisor": "Doctor Example",
        "research_categories": ["Information and Communication Technology"],
    }
    values.update(changes)
    return ProjectDocument.model_validate(values)


class FakeProvider:
    def __init__(self, available=True, response=None, error=None):
        self.available, self.response, self.error = available, response, error
        self.calls = []

    def is_available(self):
        return self.available

    def generate(self, question):
        self.calls.append(question)
        if self.error:
            raise self.error
        return self.response


class FakeRetriever:
    def __init__(self, ids=("101", "102")):
        self.ids = set(ids)
        self.calls = []

    def candidate_indices(self, scope="all", filters=None, candidate_project_ids=None):
        eligible = set(self.ids)
        if candidate_project_ids is not None:
            eligible &= set(candidate_project_ids)
        if filters and filters.get("project_id"):
            eligible &= {filters["project_id"]}
        return sorted(eligible)

    def search(self, query, top_k=5, scope="all", filters=None, *, candidate_project_ids=None):
        self.calls.append({"query": query, "scope": scope, "filters": filters,
                           "candidate_project_ids": candidate_project_ids})
        eligible = self.candidate_indices(scope, filters, candidate_project_ids)
        return [{"rank": index + 1, "item_type": "research_project", "project_id": project_id,
                 "title": f"AI Project {project_id}", "source_url": f"https://example.org/{project_id}",
                 "primary_supervisor": "Doctor Example", "provenance": [{"project_id": project_id}]}
                for index, project_id in enumerate(eligible[:top_k])]


def llm_plan(**changes):
    payload = {
        "method": "graph", "scope": "projects", "search_query": "supervisor",
        "degree_type": None, "student_type": None, "location": None, "funding_status": None,
        "status": None, "research_category": None, "supervisor": None, "project_id": "101",
        "graph_operation": "project_by_id", "intent": "project_lookup", "confidence": 0.9,
        "planner_method": "llm",
    }
    payload.update(changes)
    return FakeProvider(response=json.dumps(payload))


def test_reasoning_plan_schema_and_controlled_operation_validation():
    plan = ReasoningPlan(method="graph", scope="projects", search_query="12259", project_id="12259",
                         graph_operation="project_by_id", intent="lookup", planner_method="fallback")
    assert plan.project_filters() == {"project_id": "12259"}
    with pytest.raises(ValidationError):
        ReasoningPlan(method="graph", scope="projects", search_query="x", graph_operation="arbitrary_sparql",
                      intent="lookup", planner_method="llm")
    with pytest.raises(ValidationError):
        ReasoningPlan(method="retrieval", scope="general", search_query=" ", intent="lookup",
                      planner_method="fallback")


@pytest.mark.parametrize("question,method,scope,operation", [
    ("What English score do I need for a PhD?", "retrieval", "general", None),
    ("What documents do I need when applying?", "retrieval", "general", None),
    ("Find AI and machine learning PhD projects.", "retrieval", "projects", None),
    ("I am an international student looking for funded ICT PhD projects.", "hybrid_graph", "projects", "multi_constraint_projects"),
    ("Show me funded AI research opportunities in Hobart for international students.", "hybrid_graph", "projects", "multi_constraint_projects"),
    ("Who supervises project 12259?", "graph", "projects", "project_by_id"),
    ("Which supervisors supervise more than one advertised project?", "graph", "projects", "supervisors_with_multiple_projects"),
    ("Which supervisors have funded ICT projects accepting international students?", "graph", "projects", "supervisors_with_funded_ict_international_projects"),
    ("How many open projects are available in each research category?", "graph", "projects", "open_projects_by_category"),
    ("Which research categories contain both PhD and Master by Research projects?", "graph", "projects", "categories_with_both_degree_types"),
    ("Which supervisors work across more than one research category?", "graph", "projects", "supervisors_across_multiple_categories"),
    ("Which ICT supervisors work in artificial intelligence?", "graph", "supervisors", "ict_supervisors_by_research_field"),
    ("How many open funded international PhD projects are in ICT?", "graph", "projects", "count_open_funded_international_ict_projects"),
])
def test_fallback_broad_probe_routing(question, method, scope, operation):
    plan = fallback_reasoning_plan(question)
    assert (plan.method, plan.scope, plan.graph_operation) == (method, scope, operation)


@pytest.mark.parametrize("question,query_fragment", [
    ("Is there a research project about phishing detection?", "phishing detection"),
    ("Are there projects on marine robotics?", "marine robotics"),
    ("Find research opportunities in renewable energy.", "renewable energy"),
    ("Show me projects related to coastal erosion.", "coastal erosion"),
])
def test_generic_project_discovery_wording_uses_project_scope(question, query_fragment):
    plan = fallback_reasoning_plan(question)
    assert plan.scope == "projects"
    assert query_fragment in plan.search_query.casefold()


def test_llm_general_scope_is_rejected_for_clear_project_discovery_question():
    payload = {
        "method": "retrieval", "scope": "general", "search_query": "phishing detection",
        "degree_type": None, "student_type": None, "location": None, "funding_status": None,
        "status": None, "research_category": None, "supervisor": None, "project_id": None,
        "graph_operation": None, "intent": "project_search", "confidence": 0.9,
        "planner_method": "llm",
    }
    planner = ReasoningPlanner(FakeProvider(response=json.dumps(payload)))
    plan = planner.plan("Is there a research project about phishing detection?")
    assert plan.scope == "projects"
    assert plan.planner_method == "fallback"
    assert "project-discovery scope" in planner.last_llm_error


@pytest.mark.parametrize("question,expected", [
    ("Show projects supervised by Soonja Yeom", "Soonja Yeom"),
    ("List projects under supervisor Soonja Yeom", "Soonja Yeom"),
    ("What projects does Dr Soonja Yeom supervise?", "Dr Soonja Yeom"),
    ("Research projects by Quan Bai", "Quan Bai"),
])
def test_supervisor_project_questions_extract_name_and_use_graph(question, expected):
    plan = fallback_reasoning_plan(question)
    assert plan.method == "graph"
    assert plan.scope == "projects"
    assert plan.graph_operation == "projects_by_supervisor"
    assert plan.supervisor == expected


def test_supervisor_query_end_to_end_uses_graph_results():
    graph = build_graph([project("101", primary_supervisor="Doctor Soonja Yeom", title="Soonja Project")])
    router = ReasoningRouter(ReasoningPlanner(FakeProvider(available=False)), FakeRetriever(), graph)
    evidence = router.route("list the projects under the supervisor soonja yeom")
    assert evidence["reasoning_method"] == "graph"
    assert evidence["graph_operation_used"] == "projects_by_supervisor"
    assert [row["project_id"] for row in evidence["graph_result"]] == ["101"]


@pytest.mark.parametrize("question", ["Quan Bai projects", "Dr Quan Bai projects", "SOONJA YEOM projects"])
def test_exact_full_supervisor_name_overrides_broad_project_keyword_ranking(question):
    graph = build_graph([
        project("101", primary_supervisor="Doctor Quan Bai"),
        project("102", primary_supervisor="Doctor Soonja Yeom"),
    ])
    router = ReasoningRouter(ReasoningPlanner(FakeProvider(available=False)), FakeRetriever(), graph)
    evidence = router.route(question)
    assert evidence["graph_operation_used"] == "projects_by_supervisor"
    expected = "101" if "quan" in question.casefold() else "102"
    assert [row["project_id"] for row in evidence["graph_result"]] == [expected]


def test_shared_first_name_requests_clarification_instead_of_mixing_projects():
    graph = build_graph([
        project("101", primary_supervisor="Doctor Quan Bai"),
        project("102", primary_supervisor="Associate Professor Quan Huynh"),
    ])
    router = ReasoningRouter(ReasoningPlanner(FakeProvider(available=False)), FakeRetriever(), graph)
    evidence = router.route("Quan projects")
    assert evidence["supervisor_ambiguity"] == ["Associate Professor Quan Huynh", "Doctor Quan Bai"]
    assert evidence["ranked_retrieval_evidence"] == []


def test_retrieval_routing_calls_hybrid_for_general_scope():
    question = "What English score do I need for a PhD?"
    retriever = FakeRetriever()
    router = ReasoningRouter(ReasoningPlanner(FakeProvider(available=False)), retriever, Graph())
    evidence = router.route(question)
    assert evidence["reasoning_method"] == "retrieval"
    assert retriever.calls[0]["scope"] == "general"
    assert evidence["tool_trace"][-1]["step"] == "hybrid_retrieval"


def test_graph_route_dispatch_and_source_metadata():
    graph = build_graph([project("101")])
    router = ReasoningRouter(ReasoningPlanner(llm_plan()), FakeRetriever(), graph)
    evidence = router.route("Who supervises project 101?")
    assert evidence["reasoning_method"] == "graph"
    assert evidence["graph_operation_used"] == "project_by_id"
    assert evidence["graph_result"]["primary_supervisor"] == "Doctor Example"
    assert evidence["source_urls"] == ["https://example.org/101"]
    assert evidence["project_ids"] == ["101"]
    assert evidence["tool_trace"][-1]["step"] == "sparql_query"


def test_hybrid_graph_graph_ids_constrain_ranking_and_metadata_survives():
    graph = build_graph([project("101"), project("102", location="Launceston")])
    payload = {
        "method": "hybrid_graph", "scope": "projects", "search_query": "AI",
        "degree_type": "PhD", "student_type": "International", "location": "Hobart",
        "funding_status": "funded", "status": None,
        "research_category": "Information and Communication Technology", "supervisor": None,
        "project_id": None, "graph_operation": "multi_constraint_projects",
        "intent": "constrained_project_discovery", "confidence": 0.9, "planner_method": "llm",
    }
    provider = FakeProvider(response=json.dumps(payload))
    retriever = FakeRetriever()
    router = ReasoningRouter(ReasoningPlanner(provider), retriever, graph)
    evidence = router.route("Find funded AI PhD projects for international students in Hobart")
    assert evidence["candidate_count_before"] == 2
    assert evidence["candidate_count_after"] == 1
    assert retriever.calls[-1]["candidate_project_ids"] == {"101"}
    assert [item["project_id"] for item in evidence["ranked_retrieval_evidence"]] == ["101"]
    assert evidence["source_urls"] == ["https://example.org/101"]
    assert evidence["tool_trace"][-1]["all_results_within_graph_candidates"] is True


def test_no_invented_filters_and_invalid_json_or_unavailable_uses_fallback():
    invented = llm_plan(location="Hobart")
    planner = ReasoningPlanner(invented)
    result = planner.plan("Find funded AI PhD projects")
    assert result.planner_method == "fallback"
    assert result.location is None
    assert planner.last_llm_error
    assert ReasoningPlanner(FakeProvider(response="not-json")).plan(
        "Find funded AI PhD projects").planner_method == "fallback"
    assert ReasoningPlanner(FakeProvider(available=False)).plan(
        "Find funded AI PhD projects").planner_method == "fallback"


def test_hybrid_graph_plan_must_use_project_scope():
    provider = llm_plan(method="hybrid_graph", scope="general", student_type="International",
                        funding_status="funded", degree_type=None, project_id=None,
                        graph_operation="multi_constraint_projects")
    plan = ReasoningPlanner(provider).plan("Find funded AI projects for international students")
    assert plan.planner_method == "fallback"
    assert plan.scope == "projects"


def test_controlled_dispatch_rejects_unknown_operation_and_zero_results():
    graph = Graph()
    fake_plan = ReasoningPlan.model_construct(method="graph", scope="projects", search_query="x",
                                              graph_operation="arbitrary", intent="bad", planner_method="llm")
    with pytest.raises(ValueError, match="Unsupported graph operation"):
        execute_graph_operation(graph, fake_plan)
    outcome = ReasoningRouter(ReasoningPlanner(FakeProvider(available=False)), FakeRetriever(), graph).route(
        "Who supervises project 9999?")
    assert outcome["graph_result"] is None
    assert outcome["candidate_count_after"] == 0
