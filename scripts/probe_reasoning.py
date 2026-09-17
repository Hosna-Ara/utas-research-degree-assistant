"""Probe graph-aware reasoning routes using the 12 agreed evaluation questions."""

import json
from pathlib import Path
import re
import sys

from rdflib import Graph

from utas_research_assistant.config import OLLAMA_MODEL
from utas_research_assistant.graph.queries import (
    categories_with_phd_and_masters_by_research,
    count_open_projects_by_category,
    supervisors_across_multiple_categories,
    supervisors_with_funded_ict_international_projects,
    supervisors_with_multiple_projects,
)
from utas_research_assistant.query.planner import ReasoningPlanner, normalize_reasoning_payload
from utas_research_assistant.retrieval.filters import canonical
from utas_research_assistant.query.reasoning_router import ReasoningRouter
from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.hybrid import HybridRetriever
from utas_research_assistant.retrieval.semantic import SemanticRetriever

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
OUTPUT = PROCESSED / "reasoning_probe_results.json"
DIAGNOSTICS_OUTPUT = PROCESSED / "planner_diagnostics.json"
QUESTIONS = [
    "What English score do I need for a PhD?",
    "What documents do I need when applying?",
    "Find AI and machine learning PhD projects.",
    "I am an international student looking for funded ICT PhD projects.",
    "Show me funded AI research opportunities in Hobart for international students.",
    "Who supervises project 12259?",
    "Which supervisors supervise more than one advertised project?",
    "Which supervisors have funded ICT projects accepting international students?",
    "How many open projects are available in each research category?",
    "Which research categories contain both PhD and Master by Research projects?",
    "Which supervisors work across more than one research category?",
    "How many open funded international PhD projects are in ICT?",
]
EXPECTED_PLANS = [
    ("retrieval", "general", None, {"degree_type": "PhD"}),
    ("retrieval", "general", None, {}),
    ("retrieval", "projects", None, {"degree_type": "PhD"}),
    ("hybrid_graph", "projects", "multi_constraint_projects", {
        "degree_type": "PhD", "student_type": "International", "funding_status": "funded",
        "research_category": "Information and Communication Technology",
    }),
    ("hybrid_graph", "projects", "multi_constraint_projects", {
        "student_type": "International", "location": "Hobart", "funding_status": "funded",
    }),
    ("graph", "projects", "project_by_id", {"project_id": "12259"}),
    ("graph", "projects", "supervisors_with_multiple_projects", {}),
    ("graph", "projects", "supervisors_with_funded_ict_international_projects", {
        "student_type": "International", "funding_status": "funded",
        "research_category": "Information and Communication Technology",
    }),
    ("graph", "projects", "open_projects_by_category", {"status": "Applications open"}),
    ("graph", "projects", "categories_with_both_degree_types", {}),
    ("graph", "projects", "supervisors_across_multiple_categories", {}),
    ("graph", "projects", "count_open_funded_international_ict_projects", {
        "degree_type": "PhD", "student_type": "International", "funding_status": "funded",
        "status": "Applications open", "research_category": "Information and Communication Technology",
    }),
]
PREVIOUS_QUESTION_FOR = {
    QUESTIONS[2]: "Find funded ICT PhD projects accepting international students.",
    QUESTIONS[6]: "Which supervisors supervise more than one advertised project?",
    QUESTIONS[7]: "Which supervisors have funded ICT projects accepting international students?",
    QUESTIONS[8]: "How many open projects are available in each research category?",
    QUESTIONS[9]: "Which research categories contain both PhD and Master by Research projects?",
    QUESTIONS[10]: "Which supervisors have projects across more than one research category?",
    QUESTIONS[11]: "How many open funded international PhD projects are in ICT?",
}


def previous_graph_comparison(question: str, operation: str | None, result, previous: dict) -> dict:
    prior_question = PREVIOUS_QUESTION_FOR.get(question)
    prior = next((row for row in previous.get("evaluations", [])
                  if row.get("question") == prior_question), None)
    if not prior or result is None:
        return {"available": False, "match": None, "fields_compared": []}
    old = prior["graph_result_summary"]
    fields = {}
    if operation == "supervisors_with_multiple_projects":
        current = {"entity_count": len(result), "sample": result[:5]}
        keys = ["entity_count", "sample"]
    elif operation == "supervisors_with_funded_ict_international_projects":
        current = {"supervisor_count": len(result), "sample": result[:5]}
        keys = ["supervisor_count", "sample"]
    elif operation == "open_projects_by_category":
        current = {"category_count": len(result), "counts_by_category": result}
        keys = ["category_count", "counts_by_category"]
    elif operation == "categories_with_both_degree_types":
        current = {"category_count": len(result), "categories": result}
        keys = ["category_count", "categories"]
    elif operation == "supervisors_across_multiple_categories":
        current = {"supervisor_count": len(result), "sample": result[:5]}
        keys = ["supervisor_count", "sample"]
    elif operation == "multi_constraint_projects" and question == QUESTIONS[2]:
        ids = sorted(row["project_id"] for row in result)
        current = {"matching_project_count": len(result), "sample_project_ids": ids[:10]}
        keys = ["matching_project_count", "sample_project_ids"]
    elif operation == "count_open_funded_international_ict_projects":
        ids = sorted(row["project_id"] for row in result.get("projects", []))
        current = {"matching_project_count": result["count"], "sample_project_ids": ids[:10]}
        keys = ["matching_project_count", "sample_project_ids"]
    else:
        return {"available": False, "match": None, "fields_compared": []}
    for key in keys:
        old_key = "entity_count" if key == "entity_count" else key
        if old_key in old:
            fields[key] = current[key] == old[old_key]
    return {"available": bool(fields), "match": all(fields.values()) if fields else None,
            "fields_compared": fields}


def concise_result(evidence: dict) -> str:
    ranked = evidence["ranked_retrieval_evidence"]
    graph_result = evidence["graph_result"]
    if ranked:
        return "; ".join(f"{item['title']}" + (f" [{item['project_id']}]" if item.get("project_id") else "")
                          for item in ranked[:2])
    if isinstance(graph_result, dict) and "count" in graph_result:
        return f"{graph_result['count']} matching projects"
    if isinstance(graph_result, list):
        samples = []
        for row in graph_result[:2]:
            if isinstance(row, dict):
                samples.append(row.get("supervisor") or row.get("category") or row.get("title") or str(row))
            else:
                samples.append(str(row))
        return f"{len(graph_result)} graph results: " + "; ".join(samples)
    if isinstance(graph_result, dict):
        return graph_result.get("title") or graph_result.get("supervisor") or "Project found"
    return "No matching evidence"


def search_query_quality_issue(question: str, query: str, method: str) -> bool:
    if method == "graph":
        return False
    stop = {"what", "how", "who", "do", "does", "is", "are", "i", "me", "my", "for", "in",
            "the", "a", "an", "and", "or", "to", "of", "when", "which", "with", "on", "need",
            "show", "find", "looking", "available", "required", "projects", "project"}
    question_terms = {t for t in re.findall(r"[a-z0-9]+", question.casefold()) if len(t) > 1 and t not in stop}
    query_terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
    return bool(question_terms) and not bool(question_terms & query_terms)


def constraint_metrics(diagnostics: list[dict]) -> dict:
    expected_total = correct_total = actual_total = invented_total = exact_questions = 0
    for diagnostic, expected_row in zip(diagnostics, EXPECTED_PLANS):
        expected = expected_row[3]
        parsed = diagnostic.get("parsed_json")
        normalized = normalize_reasoning_payload(parsed) if isinstance(parsed, dict) else {}
        actual = {field: normalized.get(field) for field in (
            "degree_type", "student_type", "location", "funding_status", "status",
            "research_category", "supervisor", "project_id",
        ) if normalized.get(field) is not None}
        matched = 0
        for field, value in expected.items():
            expected_total += 1
            if field in actual and canonical(field, actual[field]) == canonical(field, value):
                matched += 1
        correct_total += matched
        actual_total += len(actual)
        invented_total += len(diagnostic.get("rejected_constraints", []))
        exact_questions += actual == expected
    return {
        "expected_constraint_count": expected_total,
        "correctly_extracted_constraint_count": correct_total,
        "constraint_extraction_accuracy": round(correct_total / expected_total, 4) if expected_total else 1.0,
        "constraint_extraction_precision": round(correct_total / actual_total, 4) if actual_total else 1.0,
        "questions_with_exact_constraint_sets": exact_questions,
        "invented_constraint_count": invented_total,
    }


def main() -> int:
    try:
        corpus = load_corpus(PROCESSED)
        graph = Graph().parse(PROCESSED / "utas_research_graph.ttl", format="turtle")
        planner = ReasoningPlanner()
        model_available = planner.llm_available
        router = ReasoningRouter(planner, HybridRetriever(
            corpus, None, SemanticRetriever(corpus, PROCESSED)), graph)
        previous = json.loads((PROCESSED / "graph_value_evaluation.json").read_text(encoding="utf-8"))
        evaluations = []
        diagnostics = []
        for index, (question, expected_plan) in enumerate(zip(QUESTIONS, EXPECTED_PLANS), 1):
            evidence = router.route(question)
            diagnostic = dict(planner.last_diagnostics)
            diagnostic["expected_broad_routing_category"] = {
                "method": expected_plan[0], "scope": expected_plan[1], "graph_operation": expected_plan[2],
            }
            quality_issue = search_query_quality_issue(
                question, diagnostic.get("parsed_json", {}).get("search_query", "")
                if isinstance(diagnostic.get("parsed_json"), dict) else "",
                diagnostic.get("parsed_json", {}).get("method", "")
                if isinstance(diagnostic.get("parsed_json"), dict) else "",
            )
            diagnostic["search_query_quality_issue"] = quality_issue
            if quality_issue and "search_query quality issue" not in diagnostic["failure_categories"]:
                diagnostic["failure_categories"].append("search_query quality issue")
            diagnostics.append(diagnostic)
            comparison = previous_graph_comparison(
                question, evidence["graph_operation_used"], evidence["graph_result"], previous,
            )
            evaluations.append({"evidence": evidence, "previous_graph_value_comparison": comparison})
            filters = json.dumps(evidence["extracted_constraints"], ensure_ascii=False) or "(none)"
            print(f"{index:02d}. {question}\n"
                  f"    method={evidence['reasoning_method']} planner={evidence['planner_method']} "
                  f"operation={evidence['graph_operation_used'] or '-'} filters={filters}\n"
                  f"    candidates={evidence['candidate_count_before']}->{evidence['candidate_count_after']} "
                  f"result={concise_result(evidence)}\n"
                  f"    qwen_accepted={diagnostic['qwen_plan_accepted']} "
                  f"rejection={diagnostic['reason_for_fallback'] or '-'}")
            if comparison["available"]:
                print(f"    previous graph evaluation match={comparison['match']} ({comparison['fields_compared']})")
            print(flush=True)
        report = {
            "model_name": OLLAMA_MODEL,
            "local_model_available": model_available,
            "local_model_used": any(row["evidence"]["planner_method"] == "llm" for row in evaluations),
            "llm_plans_accepted_count": sum(row["evidence"]["planner_method"] == "llm" for row in evaluations),
            "evaluation_method": "local prompt-driven planner; controlled local SPARQL; no answer generation",
            "question_count": len(evaluations),
            "method_counts": {method: sum(row["evidence"]["reasoning_method"] == method for row in evaluations)
                              for method in ("retrieval", "graph", "hybrid_graph")},
            "correct_broad_routing_count": sum(
                row["evidence"]["reasoning_method"] == expected[0]
                and row["evidence"]["scope"] == expected[1]
                for row, expected in zip(evaluations, EXPECTED_PLANS)
            ),
            "correct_broad_routing_rate": round(sum(
                row["evidence"]["reasoning_method"] == expected[0]
                and row["evidence"]["scope"] == expected[1]
                for row, expected in zip(evaluations, EXPECTED_PLANS)
            ) / len(QUESTIONS), 4),
            "constraint_metrics": constraint_metrics(diagnostics),
            "fallback_questions": [row["evidence"]["original_question"] for row in evaluations
                                   if row["evidence"]["planner_method"] == "fallback"],
            "failure_category_counts": {
                category: sum(category in diagnostic["failure_categories"] for diagnostic in diagnostics)
                for category in (
                    "invalid JSON", "missing method", "missing graph_operation", "omitted explicit constraint",
                    "invented constraint", "incompatible method and scope", "unsupported graph operation",
                    "search_query quality issue", "other",
                )
            },
            "planner_diagnostics": diagnostics,
            "previous_graph_comparisons": {
                "compared": sum(row["previous_graph_value_comparison"]["available"] for row in evaluations),
                "passed": sum(row["previous_graph_value_comparison"]["match"] is True for row in evaluations),
                "failed": sum(row["previous_graph_value_comparison"]["match"] is False for row in evaluations),
            },
            "results": evaluations,
        }
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Reasoning probe failed: {exc}", file=sys.stderr)
        return 1
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    DIAGNOSTICS_OUTPUT.write_text(json.dumps({
        "model_name": report["model_name"],
        "local_model_available": report["local_model_available"],
        "diagnostic_count": len(diagnostics),
        "failure_category_counts": report["failure_category_counts"],
        "probes": diagnostics,
    }, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"Ollama model available: {report['local_model_available']}; used: {report['local_model_used']}")
    print(f"Accepted LLM plans: {report['llm_plans_accepted_count']}/{report['question_count']}")
    print(f"Fallback used for {len(report['fallback_questions'])} question(s).")
    print(f"Broad-route accuracy: {report['correct_broad_routing_count']}/{report['question_count']}")
    print(f"Constraint extraction: {report['constraint_metrics']}")
    print(f"Failure categories: {report['failure_category_counts']}")
    print(f"Previous graph-value checks: {report['previous_graph_comparisons']}")
    print(f"Planner diagnostics: {DIAGNOSTICS_OUTPUT.relative_to(ROOT)}")
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
