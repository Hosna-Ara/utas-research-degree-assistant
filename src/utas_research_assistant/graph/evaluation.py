"""Small deterministic comparison helpers for graph-value evaluation."""


def compare_sets(left, right) -> dict:
    """Compare exact values, returning stable, JSON-ready difference details."""
    left_set, right_set = set(left), set(right)
    return {
        "comparison_type": "set", "match": left_set == right_set,
        "left_count": len(left_set), "right_count": len(right_set),
        "only_left": sorted(left_set - right_set, key=str),
        "only_right": sorted(right_set - left_set, key=str),
    }


def compare_counts(left: int, right: int) -> dict:
    return {"comparison_type": "count", "match": left == right,
            "left_count": left, "right_count": right, "difference": left - right}


def summarize_evaluations(evaluations: list[dict]) -> dict:
    comparisons = [row["agreement"] for row in evaluations if row.get("agreement") is not None]
    return {
        "total_evaluation_questions": len(evaluations),
        "number_best_handled_by_retrieval": sum(row["recommended_method"] == "retrieval" for row in evaluations),
        "number_best_handled_by_graph": sum(row["recommended_method"] == "graph" for row in evaluations),
        "number_where_both_contribute": sum(row["recommended_method"] == "hybrid_graph" for row in evaluations),
        "exact_comparison_pass_count": sum(item["match"] for item in comparisons),
        "exact_comparison_fail_count": sum(not item["match"] for item in comparisons),
    }
