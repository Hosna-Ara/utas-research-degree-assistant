import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from utas_research_assistant.evaluation import (
    EvaluationRecord, aggregate_evaluation, exact_count_check, hit_at_k, method_matches, ranking_metrics,
)


def test_evaluation_schema_validation():
    record = EvaluationRecord(question_id="Q1", question="English scores?", query_type="general",
                              expected_method="retrieval", actual_method="retrieval", planner_method="llm",
                              tool_invoked=False, retrieval_hit_at_5=True, citation_valid=True,
                              grounding_valid=True, deterministic_fact_check=True,
                              manual_review_required=False, generation_mode="Qwen-generated", pass_fail="pass")
    assert record.question_id == "Q1"
    with pytest.raises(ValidationError):
        EvaluationRecord(question_id="Q1", question="Q", query_type="general", expected_method="retrieval",
                         generation_mode="mystery", pass_fail="pass")


def test_hit_at_k_and_ranking_metrics():
    ranked = ["a", "b", "target", "d"]
    assert hit_at_k(ranked, {"target"}, 1) is False
    assert hit_at_k(ranked, {"target"}, 3) is True
    assert ranking_metrics(ranked, {"target"}) == {
        "hit_at_1": False, "hit_at_3": True, "hit_at_5": True,
        "expected_overlap_at_5": 1, "first_relevant_rank": 3,
    }
    with pytest.raises(ValueError):
        hit_at_k(ranked, {"target"}, 0)


def test_reasoning_method_comparison_accepts_configured_alternatives():
    assert method_matches("graph", ["hybrid_graph", "graph"])
    assert not method_matches("retrieval", ["hybrid_graph", "graph"])


def test_deterministic_count_comparison():
    assert exact_count_check(17, 17)
    assert not exact_count_check(16, 17)
    assert not exact_count_check(None, 17)


def test_evaluation_summary_aggregates_citation_grounding_and_manual_review():
    rows = [
        {"pass_fail": "pass", "routing_correct": True, "tool_expected": False, "citation_valid": True,
         "grounding_valid": True, "deterministic_fact_check": True, "manual_review_required": False,
         "planner_method": "llm", "generation_mode": "Qwen-generated"},
        {"pass_fail": "manual review", "routing_correct": True, "tool_expected": True, "tool_correct": True,
         "citation_valid": False, "grounding_valid": False, "deterministic_fact_check": None, "manual_review_required": True,
         "planner_method": "fallback", "generation_mode": "extractive fallback"},
    ]
    result = aggregate_evaluation(rows, [{"hit_at": True, "hit_at_1": True, "hit_at_3": True, "hit_at_5": True}],
                                  [{"handled_safely": True}])
    assert result["total_baseline_questions"] == 2
    assert result["baseline_questions_passed"] == 1
    assert result["citation_validity_rate"] == 0.5
    assert result["grounding_validation_rate"] == 0.5
    assert result["manual_review_count"] == 1
    assert result["qwen_planner_usage_rate"] == 0.5
    assert result["known_limitations_handled_safely"] == 1
    assert result["deterministic_fact_check_pass_rate"] == 1.0


def test_manual_review_question_does_not_claim_automatic_pass():
    row = EvaluationRecord(question_id="Q2", question="What documents?", query_type="general",
                           expected_method="retrieval", actual_method="retrieval", planner_method="llm",
                           tool_invoked=False, citation_valid=True, grounding_valid=True,
                           deterministic_fact_check=None, manual_review_required=True,
                           generation_mode="extractive fallback", pass_fail="manual review")
    assert row.pass_fail == "manual review"
    assert row.manual_review_required


def test_known_limitations_have_manual_answer_and_source_placeholders():
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / "data/evaluation/known_limitations.json").read_text(encoding="utf-8"))
    assert [row["limitation_id"] for row in rows] == ["F1", "F2", "F3"]
    assert all(row["manual_answer"] == "TO BE VERIFIED MANUALLY" for row in rows)
    assert all(row["manual_source"] == "TO BE VERIFIED MANUALLY" for row in rows)
