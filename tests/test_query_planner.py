import json

import pytest
from pydantic import ValidationError

from utas_research_assistant.query.models import RetrievalPlan
from utas_research_assistant.query.planner import QueryPlanner, fallback_plan
from utas_research_assistant.query.router import RetrievalRouter


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


def test_plan_schema_and_validation():
    plan = RetrievalPlan(scope="projects", search_query="AI", intent="discovery",
                         planner_method="fallback", confidence=0.7, degree_type="PhD")
    assert plan.project_filters() == {"degree_type": "PhD"}
    assert plan.scope == "projects"
    with pytest.raises(ValidationError):
        RetrievalPlan(scope="project", search_query="AI", intent="x", planner_method="fallback")
    with pytest.raises(ValidationError):
        RetrievalPlan(scope="general", search_query=" ", intent="x", planner_method="llm")
    with pytest.raises(ValidationError):
        RetrievalPlan(scope="all", search_query="x", intent="x", planner_method="llm", confidence=2)


@pytest.mark.parametrize("question,scope,field,value", [
    ("What English score for a PhD?", "general", "degree_type", "PhD"),
    ("Show funded AI PhD projects for international students in Hobart", "projects", "degree_type", "PhD"),
    ("I am an international student looking for a funded ICT PhD", "projects", "student_type", "International"),
    ("Master by Research projects in Launceston", "projects", "location", "Launceston"),
    ("No stipend projects in Sydney with applications open", "projects", "funding_status", "no_stipend"),
    ("Find projects with applications open in Hobart", "projects", "status", "Applications open"),
    ("Who supervises project 12259?", "projects", "project_id", "12259"),
    ("What scholarships are available?", "general", None, None),
])
def test_fallback_explicit_constraints_and_scope(question, scope, field, value):
    plan = fallback_plan(question)
    assert plan.scope == scope
    if field:
        assert getattr(plan, field) == value
    assert plan.planner_method == "fallback"


def test_fallback_does_not_invent_constraints():
    plan = fallback_plan("I am exploring computational marine ecology in Tasmania")
    assert plan.project_filters() == {}
    assert plan.search_query == "exploring computational marine ecology Tasmania"
    assert plan.scope == "general"


def test_valid_local_llm_response_is_used():
    response = json.dumps({"scope": "projects", "search_query": "AI", "degree_type": "PhD",
                           "student_type": "International", "location": "Hobart", "funding_status": "funded",
                           "status": None, "research_category": None, "supervisor": None, "project_id": None,
                           "intent": "project_discovery", "confidence": 0.9, "planner_method": "llm"})
    provider = FakeProvider(response=response)
    plan = QueryPlanner(provider).plan("Show me funded AI PhD projects for international students in Hobart")
    assert plan.planner_method == "llm"
    assert plan.search_query == "AI"
    assert plan.project_filters()["student_type"] == "International"
    assert provider.calls


@pytest.mark.parametrize("provider", [
    FakeProvider(available=False), FakeProvider(response="not json"),
    FakeProvider(response=json.dumps({"scope": "wrong", "search_query": "x"})),
    FakeProvider(available=True, error=RuntimeError("offline")),
])
def test_unavailable_or_invalid_llm_uses_fallback(provider):
    planner = QueryPlanner(provider)
    result = planner.plan("Find funded AI PhD projects in Hobart")
    assert result.planner_method == "fallback"
    assert result.scope == "projects"
    assert result.funding_status == "funded"
    assert result.degree_type == "PhD"
    assert result.location == "Hobart"
    assert planner.last_llm_error


class FakeHybrid:
    def __init__(self):
        self.calls = []

    def candidate_indices(self, scope, filters=None):
        return list(range(8 if not filters else 2))

    def search(self, query, top_k, scope, filters):
        self.calls.append((query, top_k, scope, filters))
        if filters and filters.get("project_id") == "missing":
            return []
        return [{"rank": 1, "title": "A UTAS result", "project_id": "12259",
                 "primary_supervisor": "Doctor Example", "provenance": [{"source": "saved"}],
                 "rrf_score": 0.03}]


def test_router_passes_scope_filters_and_preserves_evidence():
    planner = QueryPlanner(FakeProvider(available=False))
    hybrid = FakeHybrid()
    outcome = RetrievalRouter(planner, hybrid).route(
        "Show funded AI PhD projects for international students in Hobart")
    assert outcome["plan"].scope == "projects"
    assert outcome["candidate_count_before_filters"] == 8
    assert outcome["candidate_count_after_filters"] == 2
    query, top_k, scope, filters = hybrid.calls[0]
    assert (top_k, scope) == (5, "projects")
    assert filters == {"degree_type": "PhD", "student_type": "International",
                       "location": "Hobart", "funding_status": "funded"}
    assert outcome["results"][0]["primary_supervisor"] == "Doctor Example"


def test_general_scope_does_not_apply_project_degree_filter():
    hybrid = FakeHybrid()
    outcome = RetrievalRouter(QueryPlanner(FakeProvider(available=False)), hybrid).route(
        "What English score do I need for a PhD?")
    assert outcome["plan"].degree_type == "PhD"
    assert outcome["plan"].scope == "general"
    assert hybrid.calls[0][2:] == ("general", None)
    assert outcome["candidate_count_before_filters"] == outcome["candidate_count_after_filters"]


def test_router_zero_result_handling():
    planner = QueryPlanner(FakeProvider(available=False))
    outcome = RetrievalRouter(planner, FakeHybrid()).route("Who supervises project 12259?")
    assert outcome["plan"].project_id == "12259"
    hybrid = FakeHybrid()
    hybrid.search = lambda *args: []
    outcome = RetrievalRouter(planner, hybrid).route("Who supervises project 12259?")
    assert outcome["results"] == []
    assert outcome["candidate_count_after_filters"] == 2


def test_blank_question_rejected():
    with pytest.raises(ValueError, match="must not be blank"):
        QueryPlanner(FakeProvider()).plan("  ")
