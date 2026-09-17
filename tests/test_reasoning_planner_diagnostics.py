import json

import pytest

from utas_research_assistant.query.models import ReasoningPlan
from utas_research_assistant.query.planner import (
    REASONING_SYSTEM_PROMPT,
    ReasoningPlanner,
    _reasoning_json_schema,
    normalize_reasoning_payload,
)


class FakeProvider:
    def __init__(self, response=None, available=True):
        self.response, self.available = response, available

    def is_available(self):
        return self.available

    def generate(self, question):
        return self.response


def payload(**changes):
    base = {
        "method": "retrieval", "scope": "general", "search_query": "English score",
        "degree_type": "PhD", "student_type": None, "location": None, "funding_status": None,
        "status": None, "research_category": None, "supervisor": None, "project_id": None,
        "graph_operation": None, "intent": "english_requirements", "confidence": 0.9,
        "planner_method": "llm",
    }
    return base | changes


def test_valid_structured_output_and_private_reasoning_is_not_saved():
    response = "<think>private analysis</think>\n" + json.dumps(payload())
    planner = ReasoningPlanner(FakeProvider(response))
    plan = planner.plan("What English score do I need for a PhD?")
    assert plan.planner_method == "llm"
    assert planner.last_diagnostics["qwen_plan_accepted"] is True
    assert planner.last_diagnostics["parsed_json"]["degree_type"] == "PhD"
    assert "private analysis" not in planner.last_diagnostics["raw_model_response"]


def test_missing_explicit_constraint_is_rejected_and_diagnosed():
    planner = ReasoningPlanner(FakeProvider(json.dumps(payload(degree_type=None))))
    plan = planner.plan("What English score do I need for a PhD?")
    assert plan.planner_method == "fallback"
    assert planner.last_diagnostics["missing_explicit_constraints"] == ["degree_type"]
    assert "omitted explicit constraint" in planner.last_diagnostics["failure_categories"]


def test_invented_constraint_is_rejected_and_diagnosed():
    planner = ReasoningPlanner(FakeProvider(json.dumps(payload(location="Hobart"))))
    plan = planner.plan("What English score do I need for a PhD?")
    assert plan.planner_method == "fallback"
    assert planner.last_diagnostics["rejected_constraints"] == [{
        "field": "location", "value": "Hobart", "reason": "value is not explicit in the user question",
    }]
    assert "invented constraint" in planner.last_diagnostics["failure_categories"]


def test_conservative_alias_normalization():
    normalized = normalize_reasoning_payload({
        "degree_type": "MRes", "student_type": "international student", "status": "open",
        "research_category": "ICT", "funding_status": "no stipend",
    })
    assert normalized == {
        "degree_type": "Master by Research", "student_type": "International",
        "status": "Applications open", "research_category": "Information and Communication Technology",
        "funding_status": "no_stipend",
    }
    assert normalize_reasoning_payload({"location": "Hobart"})["location"] == "Hobart"


def test_json_only_parser_rejects_markdown_wrapping():
    planner = ReasoningPlanner(FakeProvider("```json\n" + json.dumps(payload()) + "\n```"))
    assert planner.plan("What English score do I need for a PhD?").planner_method == "fallback"
    assert "invalid JSON" in planner.last_diagnostics["failure_categories"]


def test_missing_method_and_unknown_graph_operation_are_classified():
    missing_method = ReasoningPlanner(FakeProvider(json.dumps({"scope": "general"})))
    missing_method.plan("What documents do I need when applying?")
    assert "missing method" in missing_method.last_diagnostics["failure_categories"]
    unknown = payload(method="graph", scope="projects", degree_type=None,
                      graph_operation="run_arbitrary_sparql")
    unknown_planner = ReasoningPlanner(FakeProvider(json.dumps(unknown)))
    unknown_planner.plan("Who supervises project 12259?")
    assert "unsupported graph operation" in unknown_planner.last_diagnostics["failure_categories"]


def test_graph_operation_method_scope_compatibility_and_fallback():
    invalid = payload(method="graph", scope="all", degree_type=None, project_id="12259",
                      graph_operation="project_by_id", intent="project_lookup")
    planner = ReasoningPlanner(FakeProvider(json.dumps(invalid)))
    plan = planner.plan("Who supervises project 12259?")
    assert plan.planner_method == "fallback"
    assert "incompatible method and scope" in planner.last_diagnostics["failure_categories"]
    unavailable = ReasoningPlanner(FakeProvider(available=False))
    assert unavailable.plan("Who supervises project 12259?").graph_operation == "project_by_id"
    assert unavailable.last_diagnostics["reason_for_fallback"]


def test_prompt_and_structured_schema_cover_allowed_operations_and_examples():
    prompt = REASONING_SYSTEM_PROMPT
    for example in "ABCDEFG":
        assert f"{example} " in prompt
    for operation in (
        "project_by_id", "projects_by_supervisor", "multi_constraint_projects",
        "supervisors_with_multiple_projects", "supervisors_with_funded_ict_international_projects",
        "open_projects_by_category", "categories_with_both_degree_types",
        "supervisors_across_multiple_categories", "count_open_funded_international_ict_projects",
    ):
        assert operation in prompt
    schema = _reasoning_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert "JSON object" in prompt
