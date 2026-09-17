"""Run deterministic snapshot evaluation of retrieval, routing, grounding, and answers."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

from rdflib import Graph

from utas_research_assistant.config import ANSWER_MODEL, OLLAMA_MODEL
from utas_research_assistant.evaluation import aggregate_evaluation, hit_at_k
from utas_research_assistant.generation.answer_generator import AnswerGenerator
from utas_research_assistant.generation.citations import build_citation_map
from utas_research_assistant.graph.queries import get_projects_by_constraints
from utas_research_assistant.query.planner import ReasoningPlanner
from utas_research_assistant.query.reasoning_router import GRAPH_OPERATIONS, ReasoningRouter
from utas_research_assistant.retrieval.bm25 import BM25Retriever
from utas_research_assistant.retrieval.corpus import Corpus, load_corpus
from utas_research_assistant.retrieval.filters import filter_projects
from utas_research_assistant.retrieval.hybrid import HybridRetriever
from utas_research_assistant.retrieval.project_documents import ProjectDocument
from utas_research_assistant.retrieval.semantic import SemanticRetriever

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "data/evaluation"
PROCESSED = ROOT / "data/processed"
CSV_COLUMNS = ["question_id", "question", "query_type", "expected_method", "actual_method",
               "planner_method", "tool_invoked", "retrieval_hit_at_5", "citation_valid",
               "grounding_valid", "deterministic_fact_check", "manual_review_required",
               "generation_mode", "pass_fail", "notes"]
MANUAL_REVIEW_IDS = {"Q2", "Q3", "Q4", "Q5", "Q9"}
CONSTRAINT_FIELDS = ("degree_type", "student_type", "location", "funding_status", "status",
                     "research_category", "supervisor", "project_id")


def _items_for_scope(corpus: Corpus, scope: str) -> tuple[Corpus, set[int]]:
    indices = {i for i, item in enumerate(corpus.items)
               if scope == "all" or item.item_type == ("general_chunk" if scope == "general" else "research_project")}
    selected = [item for i, item in enumerate(corpus.items) if i in indices]
    return Corpus(selected, len(selected)), indices


def _ranked_hit_metrics(rows: list[dict], question: dict, expected_ids: set[str]) -> dict | None:
    if not question.get("expected_source") and not question.get("expected_project_ids") and not expected_ids:
        return None
    expected_project_ids = set(map(str, question.get("expected_project_ids", []))) | expected_ids
    target = question.get("expected_source", "").casefold()
    def matches(row):
        return ((bool(target) and target in str(row.get("title", "")).casefold())
                or str(row.get("project_id", "")) in expected_project_ids)
    flags = [matches(row) for row in rows]
    return {
        "hit_at_1": any(flags[:1]), "hit_at_3": any(flags[:3]), "hit_at_5": any(flags[:5]),
        "expected_overlap_at_5": sum(flags[:5]),
        "first_relevant_rank": next((i for i, matched in enumerate(flags, 1) if matched), None),
    }


def _rows_matching_expected_source(rows: list[dict], expected: dict) -> list[dict]:
    source = expected.get("expected_source", "").casefold()
    ids = set(map(str, expected.get("expected_project_ids", [])))
    return [row for row in rows if (source and source in str(row.get("title", "")).casefold())
            or str(row.get("project_id", "")) in ids]


def _expected_candidate_ids(question: dict, projects: list[ProjectDocument]) -> set[str]:
    constraints = question.get("expected_constraints")
    if not constraints:
        return set(map(str, question.get("expected_project_ids", [])))
    return {str(project.project_id) for project in filter_projects(projects, **constraints)}


def _all_source_urls(evidence: dict) -> set[str]:
    return set(map(str, evidence.get("source_urls", [])))


def _validate_response(question: dict, evidence: dict, response, citation_map: dict) -> dict:
    citations_valid = all(identifier in citation_map for identifier in response.citations)
    citation_sources = {row.get("citation_id") for row in response.sources}
    citations_valid = citations_valid and set(response.citations) == citation_sources
    mention_ids = set(re.findall(r"\bproject\s+#?(\d{4,})\b", response.answer, flags=re.I))
    evidence_ids = set(map(str, evidence.get("project_ids", [])))
    grounding = citations_valid and mention_ids <= evidence_ids
    graph_count_preserved = True
    if question["question_id"] == "Q11":
        graph_count_preserved = "38" in response.answer
    elif question["question_id"] == "Q12":
        graph_count_preserved = "11" in response.answer
    elif question["question_id"] == "Q13":
        graph_count_preserved = all(f"{category}: {count}" in response.answer
                                    for category, count in question["expected_category_counts"].items())
    elif question["question_id"] == "Q14":
        graph_count_preserved = len(evidence.get("graph_result") or []) == question["expected_count"]
    elif question["question_id"] == "Q15":
        graph_count_preserved = (isinstance(evidence.get("graph_result"), dict)
                                 and evidence["graph_result"].get("count") == question["expected_count"]
                                 and str(question["expected_count"]) in response.answer)
    return {"citation_valid": citations_valid, "citation_ids_map_to_evidence": citations_valid,
            "project_ids_in_answer_are_grounded": mention_ids <= evidence_ids,
            "grounding_valid": grounding, "graph_counts_preserved": graph_count_preserved,
            "answer_generated": bool(response.answer.strip()), "mentioned_project_ids": sorted(mention_ids),
            "evidence_project_ids": sorted(evidence_ids)}


def _facts_check(question: dict, evidence: dict, response) -> tuple[bool | None, dict]:
    qid = question["question_id"]
    result = evidence.get("graph_result")
    answer = response.answer
    details: dict = {}
    if qid == "Q1":
        required = question["expected_facts"]
        checks = {key: str(value) in answer for key, value in required.items()}
        details = checks
        return all(checks.values()), details
    if qid in {"Q5", "Q6"}:
        expected = set(question["expected_project_ids"])
        appeared = {str(row.get("project_id")) for row in evidence.get("ranked_retrieval_evidence", [])}
        details = {"expected_project_ids": sorted(expected), "retrieved_project_ids": sorted(appeared)}
        return bool(expected & appeared), details
    if qid in {"Q7", "Q8", "Q9"}:
        constraints = question.get("expected_constraints", {})
        actual_constraints = evidence.get("extracted_constraints", {})
        constraint_checks = {key: actual_constraints.get(key) == value for key, value in constraints.items()}
        forbidden_ok = all(actual_constraints.get(key) is None for key in question.get("forbidden_constraints", []))
        graph_ids = {str(row.get("project_id")) for row in result or [] if isinstance(row, dict) and row.get("project_id")}
        expected_ids = _expected_candidate_ids(question, _PROJECTS)
        count_ok = (len(graph_ids) == question["expected_candidate_count"]
                    if question.get("expected_candidate_count") is not None else True)
        set_ok = (graph_ids == expected_ids) if question.get("expected_candidate_count") is not None else bool(graph_ids)
        details = {"constraint_checks": constraint_checks, "forbidden_constraints_ok": forbidden_ok,
                   "actual_candidate_count": len(graph_ids), "expected_candidate_count": question.get("expected_candidate_count"),
                   "candidate_ids_match_structured_filters": set_ok}
        return all(constraint_checks.values()) and forbidden_ok and count_ok and set_ok, details
    if qid == "Q10":
        supervisor = result.get("primary_supervisor") if isinstance(result, dict) else None
        details = {"expected_supervisor": question["expected_supervisor"], "graph_supervisor": supervisor}
        return supervisor == question["expected_supervisor"] and question["expected_supervisor"] in answer, details
    if qid in {"Q11", "Q12", "Q13", "Q14"}:
        count = len(result or [])
        checks = {"result_count_matches": count == question["expected_count"]}
        if qid == "Q13":
            by_category = {row.get("category"): row.get("count") for row in result or []}
            checks["known_category_values_match"] = all(by_category.get(name) == value
                                                         for name, value in question["expected_category_counts"].items())
        elif qid == "Q14":
            checks["all_categories_listed"] = all(category in answer for category in result or [])
        elif qid == "Q11":
            checks["answer_reports_count"] = "38" in answer
        elif qid == "Q12":
            checks["answer_reports_count"] = "11" in answer
        details = checks
        return all(checks.values()), details
    if qid == "Q15":
        count = result.get("count") if isinstance(result, dict) else None
        passed = count == question["expected_count"] and str(question["expected_count"]) in answer
        details = {"actual_count": count, "expected_count": question["expected_count"]}
        return passed, details
    return None, {"reason": "No fully specified deterministic fact oracle; manual review required."}


def _generation_mode(generator: AnswerGenerator, evidence: dict) -> str:
    mode = generator.last_generation_method
    if mode == "ollama":
        return "Qwen-generated"
    if mode == "deterministic_evidence_render":
        return ("deterministic graph rendering" if evidence.get("reasoning_method") == "graph"
                else "ranked-evidence rendering")
    if mode == "extractive_or_ranked_evidence_fallback":
        return ("ranked-evidence rendering" if evidence.get("reasoning_method") == "hybrid_graph"
                or evidence.get("scope") == "projects" else "extractive fallback")
    if mode == "insufficient_evidence_fallback":
        return "insufficient-evidence fallback"
    return "not generated"


def _dataset_retrieval_scope(question: dict) -> str:
    return "general" if question["question_id"] in {"Q1", "Q2", "Q3", "Q4"} else "projects"


def _evaluate_retrieval_designs(question: dict, corpus: Corpus, semantic: SemanticRetriever,
                                hybrid: HybridRetriever, expected_ids: set[str]) -> dict:
    scope = _dataset_retrieval_scope(question)
    scoped, indices = _items_for_scope(corpus, scope)
    bm25_rows = BM25Retriever(scoped).search(question["question"], 5)
    semantic_rows = semantic.search(question["question"], 5, candidate_indices=indices)
    hybrid_rows = hybrid.search(question["question"], 5, scope=scope)
    rankings = {}
    for name, rows in (("BM25", bm25_rows), ("Semantic", semantic_rows), ("Hybrid", hybrid_rows)):
        rankings[name] = {"items": rows, "metrics": _ranked_hit_metrics(rows, question, expected_ids)}
    return {"question_id": question["question_id"], "question": question["question"],
            "retrieval_scope": scope, "expected_ids_or_sources": sorted(expected_ids), "rankings": rankings}


def _known_limitations(router: ReasoningRouter, generator: AnswerGenerator) -> list[dict]:
    limitations = json.loads((EVALUATION / "known_limitations.json").read_text(encoding="utf-8"))
    results = []
    for item in limitations:
        try:
            evidence = router.route(item["question"])
            answer = generator.generate(item["question"], evidence)
            answer_text = answer.answer.casefold()
            safe_indicators = {
                "F1": "does not include the detailed eligibility" in answer_text,
                "F2": "publication histories" in answer_text and "not included" in answer_text,
                "F3": "cannot verify live utas website changes" in answer_text,
            }
            safe = answer.insufficient_evidence and safe_indicators.get(item["limitation_id"], False)
            observed_stage = "generation"
            observed_reason = "The answer used available evidence without clearly stating that the requested detail was absent."
            if safe:
                observed_stage = "evidence_guard"
                observed_reason = "A deterministic capability check identified the missing evidence and returned a safe limitation response."
            elif evidence.get("reasoning_method") == "retrieval" and not answer.insufficient_evidence:
                observed_stage = "retrieval"
                observed_reason = "Retrieved material was unrelated to the live/current-change question and was presented as an answer."
            results.append({**item, "observed_response": answer.model_dump(), "reasoning_evidence": evidence,
                            "handled_safely": safe,
                            "observed_failure_stage": observed_stage,
                            "observed_failure_reason": observed_reason,
                            "error": generator.last_error})
        except Exception as exc:
            results.append({**item, "observed_response": None, "reasoning_evidence": None,
                            "handled_safely": False, "observed_failure_stage": "generation",
                            "observed_failure_reason": "The evaluation request raised an exception before a grounded answer was produced.",
                            "error": f"{type(exc).__name__}: {exc}"})
    output = [{**row, "manual_answer": "TO BE VERIFIED MANUALLY", "manual_source": "TO BE VERIFIED MANUALLY"}
              for row in results]
    (PROCESSED / "known_limitations_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    global _PROJECTS
    questions = json.loads((EVALUATION / "baseline_questions.json").read_text(encoding="utf-8"))
    try:
        corpus = load_corpus(PROCESSED)
        graph = Graph().parse(PROCESSED / "utas_research_graph.ttl", format="turtle")
        semantic = SemanticRetriever(corpus, PROCESSED)
        hybrid = HybridRetriever(corpus, BM25Retriever(corpus), semantic)
        router = ReasoningRouter(ReasoningPlanner(), hybrid, graph)
        answer_generator = AnswerGenerator()
        _PROJECTS = [ProjectDocument.model_validate({**item.metadata, "text": item.text})
                     for item in corpus.items if item.item_type == "research_project"]
    except Exception as exc:
        print(f"Evaluation setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    rows, retrieval_comparison, csv_rows, baseline_results = [], [], [], []
    expected_operator = {"Q7": "multi_constraint_projects", "Q8": "multi_constraint_projects",
                         "Q9": "multi_constraint_projects", "Q10": "project_by_id",
                         "Q11": "supervisors_with_multiple_projects",
                         "Q12": "supervisors_with_funded_ict_international_projects",
                         "Q13": "open_projects_by_category",
                         "Q14": "categories_with_both_degree_types",
                         "Q15": "count_open_funded_international_ict_projects"}

    for question in questions:
        try:
            evidence = router.route(question["question"])
            answer = answer_generator.generate(question["question"], evidence)
            citation_map, citation_sources = build_citation_map(evidence)
            grounding = _validate_response(question, evidence, answer, citation_map)
            fact_pass, fact_details = _facts_check(question, evidence, answer)
            expected_ids = _expected_candidate_ids(question, _PROJECTS)
            if question.get("expected_source") or question.get("expected_project_ids"):
                retrieval_targets = set(map(str, question.get("expected_project_ids", [])))
            else:
                retrieval_targets = expected_ids
            routed_hit = _ranked_hit_metrics(evidence.get("ranked_retrieval_evidence", []), question, retrieval_targets)
            routing_correct = evidence["reasoning_method"] in question["expected_method"]
            if question["question_id"] == "Q8" and evidence["reasoning_method"] not in question["expected_method"]:
                routing_correct = False
            tool_expected = question["question_id"] in expected_operator
            tool_ok = (not tool_expected or (evidence.get("graph_operation_used") == expected_operator[question["question_id"]]
                                              and evidence.get("graph_result") is not None
                                              and expected_operator[question["question_id"]] in GRAPH_OPERATIONS))
            manual = question["question_id"] in MANUAL_REVIEW_IDS
            has_error = not answer.answer or answer.insufficient_evidence and question["question_id"] not in {"Q9"}
            # Q2–Q4 can be relevant and cite evidence yet still need human semantic review.
            checks = [routing_correct, tool_ok, grounding["citation_valid"], grounding["grounding_valid"]]
            if fact_pass is not None:
                checks.append(fact_pass)
            if question.get("expected_source") or question.get("expected_project_ids") or question["question_id"] in {"Q7", "Q8"}:
                checks.append(bool(routed_hit and routed_hit["hit_at_5"]))
            if has_error:
                checks.append(False)
            pass_fail = "error" if answer_generator.last_error and not answer.answer else (
                "fail" if not all(checks) else "manual review" if manual else "pass")
            row = {
                "question_id": question["question_id"], "question": question["question"],
                "query_type": question["query_type"], "expected_method": question["expected_method"],
                "actual_method": evidence["reasoning_method"], "planner_method": evidence["planner_method"],
                "answer_model_called": answer_generator.last_model_called,
                "answer_generation_method": answer_generator.last_generation_method,
                "generation_mode": _generation_mode(answer_generator, evidence),
                "tool_invoked": evidence.get("graph_operation_used") is not None,
                "tool_operation": evidence.get("graph_operation_used"), "tool_correct": tool_ok,
                "tool_expected": tool_expected, "routing_correct": routing_correct,
                "candidate_count_before": evidence["candidate_count_before"],
                "candidate_count_after": evidence["candidate_count_after"],
                "retrieval_hit_at_5": routed_hit["hit_at_5"] if routed_hit else None,
                "retrieval_hit_metrics": routed_hit,
                "expected_source_matches": [item for item in _rows_matching_expected_source(
                    evidence.get("ranked_retrieval_evidence", []), question)],
                "citation_valid": grounding["citation_valid"], "grounding_valid": grounding["grounding_valid"],
                "grounding_checks": grounding, "citation_count": len(answer.citations),
                "citations": answer.citations, "citation_sources": citation_sources,
                "deterministic_fact_check": fact_pass, "fact_check_details": fact_details,
                "manual_review_required": manual, "pass_fail": pass_fail,
                "notes": "" if not manual else "Automated checks are limited; inspect saved answer and evidence manually.",
                "answer": answer.model_dump(), "evidence": evidence,
                "generation_error": answer_generator.last_error,
            }
            rows.append(row)
            baseline_results.append({**row, "tool_expected": tool_expected})

            comparison = _evaluate_retrieval_designs(question, corpus, semantic, hybrid, retrieval_targets)
            if question["question_id"] in {"Q7", "Q8", "Q9"}:
                actual_graph_ids = {str(item.get("project_id")) for item in evidence.get("graph_result", [])
                                    if isinstance(item, dict) and item.get("project_id")}
                comparison["graph_enhanced"] = {
                    "method": evidence["reasoning_method"], "operation": evidence["graph_operation_used"],
                    "eligible_candidate_count": len(actual_graph_ids),
                    "eligible_set_matches_structured_filter_oracle": actual_graph_ids == expected_ids,
                    "ranked_evidence": evidence.get("ranked_retrieval_evidence", []),
                    "metrics": _ranked_hit_metrics(evidence.get("ranked_retrieval_evidence", []), question, expected_ids),
                }
                comparison["hybrid_alone_vs_graph_enhanced"] = {
                    "hybrid_alone_top5": comparison["rankings"]["Hybrid"]["items"],
                    "hybrid_alone_metrics": comparison["rankings"]["Hybrid"]["metrics"],
                    "graph_enhanced_metrics": comparison["graph_enhanced"]["metrics"],
                    "observable_graph_advantage": ("Graph applies all explicit metadata constraints before ranking and returns the exact eligible candidate set; hybrid-only ranking does not enforce those constraints."
                                                    if actual_graph_ids == expected_ids else "Graph candidate set differed from the deterministic structured-filter oracle; inspect this failure."),
                }
            elif question["question_id"] in {"Q10", "Q11", "Q12", "Q13", "Q14", "Q15"}:
                comparison["hybrid_alone_vs_graph_enhanced"] = {
                    "hybrid_alone_top5": comparison["rankings"]["Hybrid"]["items"],
                    "graph_operation": evidence.get("graph_operation_used"),
                    "graph_result": evidence.get("graph_result"),
                    "observable_graph_advantage": "SPARQL returns the exact project/supervisor/category relationship or aggregate; a ranked top-five retrieval cannot establish an exhaustive set or exact count.",
                    "ranking_metrics_applicable": False,
                }
            retrieval_comparison.append(comparison)
            csv_rows.append({"question_id": row["question_id"], "question": row["question"],
                             "query_type": row["query_type"], "expected_method": "|".join(row["expected_method"]),
                             "actual_method": row["actual_method"], "planner_method": row["planner_method"],
                             "tool_invoked": row["tool_invoked"], "retrieval_hit_at_5": row["retrieval_hit_at_5"],
                             "citation_valid": row["citation_valid"], "grounding_valid": row["grounding_valid"],
                             "deterministic_fact_check": row["deterministic_fact_check"],
                             "manual_review_required": row["manual_review_required"],
                             "generation_mode": row["generation_mode"], "pass_fail": row["pass_fail"],
                             "notes": row["notes"]})
            print(f"{row['question_id']} {row['question']}\n  route={row['actual_method']}/{row['planner_method']} "
                  f"tool={row['tool_operation'] or '-'} candidates={row['candidate_count_after']} "
                  f"fact={fact_pass} hit@5={row['retrieval_hit_at_5']} mode={row['generation_mode']} "
                  f"manual={manual}\n  answer={answer.answer[:300]}\n", flush=True)
        except Exception as exc:
            row = {"question_id": question["question_id"], "question": question["question"],
                   "query_type": question["query_type"], "expected_method": question["expected_method"],
                   "actual_method": None, "planner_method": None, "tool_invoked": False,
                   "retrieval_hit_at_5": None, "citation_valid": False, "grounding_valid": False,
                   "deterministic_fact_check": False, "manual_review_required": True,
                   "generation_mode": "not generated", "pass_fail": "error",
                   "error": f"{type(exc).__name__}: {exc}", "answer": None, "evidence": None}
            rows.append(row)
            baseline_results.append({**row, "tool_expected": question["question_id"] in expected_operator})
            print(f"{question['question_id']} ERROR: {row['error']}", file=sys.stderr, flush=True)

    known = _known_limitations(router, answer_generator)
    known_out = [{key: value for key, value in row.items() if key not in {"reasoning_evidence"}} for row in known]
    retrieval_metric_rows = []
    for comparison in retrieval_comparison:
        if comparison.get("rankings", {}).get("Hybrid", {}).get("metrics") is not None:
            metrics = comparison["rankings"]["Hybrid"]["metrics"]
            retrieval_metric_rows.append({"hit_at": True, **metrics})
    summary = aggregate_evaluation(baseline_results, retrieval_metric_rows, known)
    summary["answer_model"] = ANSWER_MODEL
    summary["planner_model"] = OLLAMA_MODEL
    summary["qwen_answer_model_invocation_count"] = sum(row.get("answer_model_called", False) for row in rows)
    summary["qwen_answer_model_invocation_rate"] = round(summary["qwen_answer_model_invocation_count"] / len(rows), 4) if rows else None
    summary["qwen_answer_generation_usage_rate"] = round(sum(row.get("generation_mode") == "Qwen-generated" for row in rows) / len(rows), 4) if rows else None
    summary["manual_review_question_ids"] = [row["question_id"] for row in rows if row.get("manual_review_required")]
    summary["known_limitations_safe_ids"] = [row["limitation_id"] for row in known if row.get("handled_safely")]
    design_metrics = {}
    for design in ("BM25", "Semantic", "Hybrid"):
        applicable = [row["rankings"][design]["metrics"] for row in retrieval_comparison
                      if row.get("rankings", {}).get(design, {}).get("metrics") is not None]
        design_metrics[design] = {metric: round(sum(bool(item[metric]) for item in applicable) / len(applicable), 4)
                                  if applicable else None for metric in ("hit_at_1", "hit_at_3", "hit_at_5")}
        design_metrics[design]["question_count"] = len(applicable)
    summary["retrieval_design_metrics"] = design_metrics
    snapshot_date = "unknown"
    manifest_path = PROCESSED / "snapshot_manifest.json"
    if manifest_path.exists():
        try:
            snapshot_date = json.loads(manifest_path.read_text(encoding="utf-8")).get("snapshot_date", "unknown")
        except (ValueError, OSError):
            pass
    inputs = [PROCESSED / name for name in ("general_chunks.json", "project_documents.json", "semantic_index.json", "utas_research_graph.ttl")]
    metadata = {"snapshot_date": snapshot_date, "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "dataset_question_count": len(questions), "baseline_question_count": len(rows),
                "input_sha256": {path.name: _sha256(path) for path in inputs if path.exists()},
                "automatic_vs_manual": "Deterministic checks are reported separately; citation validity alone is not factual validation."}
    (PROCESSED / "system_evaluation_results.json").write_text(
        json.dumps({"metadata": metadata, "results": rows}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (PROCESSED / "system_evaluation_summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (PROCESSED / "retrieval_comparison.json").write_text(
        json.dumps({"metadata": metadata, "comparisons": retrieval_comparison}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    with (PROCESSED / "evaluation_table.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\nAUTOMATICALLY VERIFIED")
    print(json.dumps({key: summary.get(key) for key in (
        "total_baseline_questions", "baseline_questions_passed", "retrieval_hit_at_1",
        "retrieval_hit_at_3", "retrieval_hit_at_5", "routing_accuracy",
        "graph_tool_invocation_accuracy", "citation_validity_rate", "grounding_validation_rate",
        "deterministic_fact_check_pass_rate", "qwen_planner_usage_rate",
        "qwen_answer_generation_usage_rate", "qwen_answer_model_invocation_rate",
        "fallback_usage_count", "errors_or_crashes")}, indent=2))
    print("REQUIRES MANUAL REVIEW")
    print(json.dumps({"manual_review_count": summary["manual_review_count"],
                      "questions": summary["manual_review_question_ids"],
                      "known_limitations_handled_safely": summary["known_limitations_handled_safely"],
                      "known_limitations": [{"id": row["limitation_id"], "handled_safely": row["handled_safely"],
                                            "failure_stage": row["failure_stage"]} for row in known]}, indent=2))
    print(f"Saved evaluation artifacts to {PROCESSED}")
    return 0 if not summary["errors_or_crashes"] else 1


_PROJECTS: list[ProjectDocument] = []


if __name__ == "__main__":
    raise SystemExit(main())
