"""Run deterministic end-to-end answer and grounding probes."""

import json
from pathlib import Path
import re
import sys

from utas_research_assistant.generation.answer_generator import AnswerGenerator
from utas_research_assistant.generation.citations import build_citation_map
from utas_research_assistant.config import ANSWER_MODEL
from ask import make_router

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
OUTPUT = PROCESSED / "answer_probe_results.json"
QUESTIONS = [
    "What English score do I need for a PhD?",
    "What documents do I need when applying?",
    "What scholarships are available for research students?",
    "Find AI and machine learning PhD projects.",
    "I am an international student looking for funded ICT PhD projects.",
    "Show me funded AI research opportunities in Hobart for international students.",
    "Who supervises project 12259?",
    "Which supervisors supervise more than one advertised project?",
    "How many open projects are available in each research category?",
    "Which research categories contain both PhD and Master by Research projects?",
    "How many open funded international PhD projects are in ICT?",
    "What is the guaranteed admission probability for an applicant with a GPA of 3.5?",
]


def grounding_checks(question: str, answer, evidence: dict) -> dict:
    citation_map, sources = build_citation_map(evidence)
    source_ids = {row["citation_id"] for row in sources}
    citations_valid = set(answer.citations) <= source_ids and all(c in citation_map for c in answer.citations)
    evidence_ids = set(map(str, evidence.get("project_ids", [])))
    cited_project_ids = set()
    for citation in answer.citations:
        project_id = citation_map[citation].get("project_id")
        if project_id:
            cited_project_ids.add(str(project_id))
    mentions = set(re.findall(r"\bproject\s+#?(\d{4,})\b", answer.answer, re.I))
    project_ids_grounded = mentions <= evidence_ids
    graph = evidence.get("graph_result")
    graph_numbers_match = True
    if evidence.get("reasoning_method") in {"graph", "hybrid_graph"}:
        # Numeric claims for known exact count operations must preserve the deterministic result.
        if isinstance(graph, dict) and "count" in graph:
            graph_numbers_match = str(graph["count"]) in answer.answer
        elif isinstance(graph, list) and graph and all(isinstance(row, dict) and "count" in row for row in graph):
            graph_numbers_match = all(
                row["category"] in answer.answer and str(row["count"]) in answer.answer
                for row in graph if row.get("category")
            )
    unsupported_handled = (answer.insufficient_evidence and
                           "does not contain enough information" in answer.answer.casefold()) if "guaranteed admission probability" in question.casefold() else None
    return {
        "answer_generated": bool(answer.answer.strip()),
        "citations_valid": citations_valid,
        "every_citation_maps_to_evidence": citations_valid,
        "project_ids_grounded": project_ids_grounded,
        "graph_numerical_answers_match": graph_numbers_match,
        "unsupported_question_handled": unsupported_handled,
        "citation_map_size": len(citation_map),
        "source_count": len(answer.sources),
    }


def main() -> int:
    if not (PROCESSED / "semantic_embeddings.npy").exists():
        print("Semantic index is missing; build embeddings first.", file=sys.stderr)
        return 1
    router = make_router(PROCESSED)
    generator = AnswerGenerator()
    results = []
    for question in QUESTIONS:
        try:
            evidence = router.route(question)
            answer = generator.generate(question, evidence)
            checks = grounding_checks(question, answer, evidence)
            results.append({"question": question, "evidence": evidence,
                            "response": answer.model_dump(), "validation": checks,
                            "answer_generation_method": generator.last_generation_method,
                            "generation_error": generator.last_error})
            print(f"{question}\n  {answer.reasoning_method}/{answer.planner_method} | "
                  f"valid citations={checks['citations_valid']} | {answer.answer[:230]}\n")
        except Exception as exc:
            results.append({"question": question, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{question}\n  ERROR: {type(exc).__name__}: {exc}\n")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    payload = {"answer_model": ANSWER_MODEL, "probe_count": len(results), "results": results,
               "summary": {
                   "answers_generated": sum(bool(row.get("response", {}).get("answer")) for row in results),
                   "citation_checks_passed": sum(bool(row.get("validation", {}).get("citations_valid")) for row in results),
                   "grounding_checks_passed": sum(all(value is not False for key, value in row.get("validation", {}).items()
                                                     if key not in {"unsupported_question_handled"}) and
                                                     row.get("validation", {}).get("unsupported_question_handled") is not False
                                                     for row in results),
                   "answer_generation_methods": {
                       method: sum(row.get("answer_generation_method") == method for row in results)
                       for method in sorted({row.get("answer_generation_method") for row in results if row.get("answer_generation_method")})
                   },
                   "errors": sum("error" in row for row in results),
                   "unsupported_question_passed": next((row.get("validation", {}).get("unsupported_question_handled")
                                                        for row in results if "guaranteed admission probability" in row["question"].casefold()), None),
               }}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Saved full probe output: {OUTPUT}")
    print(json.dumps(payload["summary"], indent=2))
    return 0 if payload["summary"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
