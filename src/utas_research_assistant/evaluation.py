"""Small deterministic helpers and schemas for snapshot system evaluation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    query_type: str
    expected_method: str
    actual_method: str | None = None
    planner_method: str | None = None
    tool_invoked: bool = False
    retrieval_hit_at_5: bool | None = None
    citation_valid: bool | None = None
    grounding_valid: bool | None = None
    deterministic_fact_check: bool | None = None
    manual_review_required: bool = False
    generation_mode: Literal["Qwen-generated", "deterministic graph rendering", "extractive fallback",
                             "ranked-evidence rendering", "insufficient-evidence fallback", "not generated"]
    pass_fail: Literal["pass", "fail", "manual review", "error"]
    notes: str = ""


def hit_at_k(ranked_ids: list[str], expected_ids: set[str] | list[str], k: int) -> bool:
    if k <= 0:
        raise ValueError("k must be positive")
    return bool(set(map(str, ranked_ids[:k])) & set(map(str, expected_ids)))


def ranking_metrics(ranked_ids: list[str], expected_ids: set[str] | list[str]) -> dict:
    expected = set(map(str, expected_ids))
    return {
        "hit_at_1": hit_at_k(ranked_ids, expected, 1),
        "hit_at_3": hit_at_k(ranked_ids, expected, 3),
        "hit_at_5": hit_at_k(ranked_ids, expected, 5),
        "expected_overlap_at_5": len(set(map(str, ranked_ids[:5])) & expected),
        "first_relevant_rank": next((i for i, value in enumerate(ranked_ids, 1) if str(value) in expected), None),
    }


def method_matches(actual: str | None, expected: str | list[str]) -> bool:
    allowed = {expected} if isinstance(expected, str) else set(expected)
    return actual in allowed


def exact_count_check(actual: int | None, expected: int) -> bool:
    return actual == expected


def aggregate_evaluation(results: list[dict], retrieval_rows: list[dict],
                         known_limitations: list[dict] | None = None) -> dict:
    total = len(results)
    applicable = [row for row in retrieval_rows if row.get("hit_at") is not None]
    denominator = len(applicable)
    def avg(metric: str):
        return round(sum(bool(row[metric]) for row in applicable) / denominator, 4) if denominator else None
    graph_expected = [row for row in results if row.get("tool_expected")]
    summary = {
        "total_baseline_questions": total,
        "baseline_questions_passed": sum(row.get("pass_fail") == "pass" for row in results),
        "routing_accuracy": round(sum(row.get("routing_correct", False) for row in results) / total, 4) if total else None,
        "graph_tool_invocation_accuracy": round(sum(row.get("tool_correct", False) for row in graph_expected) / len(graph_expected), 4) if graph_expected else None,
        "citation_validity_rate": round(sum(row.get("citation_valid", False) for row in results) / total, 4) if total else None,
        "grounding_validation_rate": round(sum(row.get("grounding_valid", False) for row in results) / total, 4) if total else None,
        "deterministic_fact_check_pass_rate": round(sum(row.get("deterministic_fact_check", row.get("fact_check")) is True for row in results if row.get("deterministic_fact_check", row.get("fact_check")) is not None) / max(1, sum(row.get("deterministic_fact_check", row.get("fact_check")) is not None for row in results)), 4) if any(row.get("deterministic_fact_check", row.get("fact_check")) is not None for row in results) else None,
        "manual_review_count": sum(row.get("manual_review_required", False) for row in results),
        "qwen_planner_usage_rate": round(sum(row.get("planner_method") == "llm" for row in results) / total, 4) if total else None,
        "qwen_answer_generation_usage_rate": round(sum(row.get("generation_mode") == "Qwen-generated" for row in results) / total, 4) if total else None,
        "generation_mode_counts": {mode: sum(row.get("generation_mode") == mode for row in results) for mode in (
            "Qwen-generated", "deterministic graph rendering", "extractive fallback",
            "ranked-evidence rendering", "insufficient-evidence fallback")},
        "fallback_usage_count": sum(row.get("generation_mode") in {
            "extractive fallback", "ranked-evidence rendering", "insufficient-evidence fallback"} for row in results),
        "known_limitations_handled_safely": sum(row.get("handled_safely", False) for row in (known_limitations or [])),
        "known_limitation_count": len(known_limitations or []),
        "errors_or_crashes": sum(row.get("pass_fail") == "error" for row in results),
        "retrieval_metric_question_count": denominator,
        "retrieval_hit_at_1": avg("hit_at_1"),
        "retrieval_hit_at_3": avg("hit_at_3"),
        "retrieval_hit_at_5": avg("hit_at_5"),
        "evaluation_note": "Snapshot-specific automated checks are not a claim of complete factual accuracy; manual review remains necessary.",
    }
    return summary
