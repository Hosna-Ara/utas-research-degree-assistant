from utas_research_assistant.graph.evaluation import compare_counts, compare_sets, summarize_evaluations


def test_set_comparison_is_exact_and_order_independent():
    same = compare_sets(["a", "b"], ["b", "a"])
    assert same["match"] is True
    assert same["only_left"] == same["only_right"] == []
    different = compare_sets(["a", "b"], ["b", "c"])
    assert different["match"] is False
    assert different["only_left"] == ["a"]
    assert different["only_right"] == ["c"]


def test_count_comparison_and_zero_result_handling():
    assert compare_counts(0, 0)["match"] is True
    assert compare_counts(0, 1)["match"] is False
    assert compare_sets([], [])["match"] is True


def test_evaluation_summary_counts_recommendation_groups_and_comparisons():
    rows = [
        {"recommended_method": "retrieval", "agreement": None},
        {"recommended_method": "graph", "agreement": {"match": True}},
        {"recommended_method": "hybrid_graph", "agreement": {"match": False}},
    ]
    summary = summarize_evaluations(rows)
    assert summary == {
        "total_evaluation_questions": 3,
        "number_best_handled_by_retrieval": 1,
        "number_best_handled_by_graph": 1,
        "number_where_both_contribute": 1,
        "exact_comparison_pass_count": 1,
        "exact_comparison_fail_count": 1,
    }


def test_empty_evaluation_summary_is_zero_safe():
    summary = summarize_evaluations([])
    assert summary["total_evaluation_questions"] == 0
    assert summary["exact_comparison_pass_count"] == 0
    assert summary["exact_comparison_fail_count"] == 0
