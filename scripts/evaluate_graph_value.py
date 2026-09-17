"""Deterministically compare graph answers with the existing local retrieval baseline."""

import json
from pathlib import Path
import sys

from rdflib import Graph

from utas_research_assistant.graph.queries import (
    categories_with_phd_and_masters_by_research,
    count_open_projects_by_category,
    get_project_by_id,
    get_projects_by_constraints,
    supervisors_across_multiple_categories,
    supervisors_with_funded_ict_international_projects,
    supervisors_with_multiple_projects,
)
from utas_research_assistant.graph.evaluation import compare_counts, compare_sets, summarize_evaluations
from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.filters import filter_projects
from utas_research_assistant.retrieval.hybrid import HybridRetriever
from utas_research_assistant.retrieval.project_documents import ProjectDocument
from utas_research_assistant.retrieval.semantic import SemanticRetriever

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
OUT = PROCESSED / "graph_value_evaluation.json"
QUESTIONS = [
    ("What English score is required for a PhD?", "retrieval"),
    ("What documents are required when applying?", "retrieval"),
    ("What scholarships are available?", "retrieval"),
    ("Find funded ICT PhD projects accepting international students.", "hybrid_graph"),
    ("Show Master by Research projects in Hobart.", "hybrid_graph"),
    ("Who supervises project 12259?", "hybrid_graph"),
    ("Which supervisors supervise more than one advertised project?", "graph"),
    ("Which supervisors have funded ICT projects accepting international students?", "graph"),
    ("How many open projects are available in each research category?", "graph"),
    ("Which research categories contain both PhD and Master by Research projects?", "graph"),
    ("Which supervisors have projects across more than one research category?", "graph"),
    ("How many open funded international PhD projects are in ICT?", "hybrid_graph"),
]
FILTERS_4 = {
    "degree_type": "PhD", "student_type": "International", "funding_status": "funded",
    "research_category": "Information and Communication Technology",
}
FILTERS_5 = {"degree_type": "Master by Research", "location": "Hobart"}
FILTERS_12 = {**FILTERS_4, "status": "Applications open"}


def _titles(results: list[dict], limit=5) -> list[dict]:
    return [{"title": row.get("title"), "project_id": row.get("project_id")} for row in results[:limit]]


def _retrieval_search(retriever, question, *, scope="projects", filters=None, top_k=5):
    return retriever.search(question, top_k=top_k, scope=scope, filters=filters)


def build_evaluation() -> dict:
    if not (PROCESSED / "utas_research_graph.ttl").is_file():
        raise FileNotFoundError(f"Run scripts/build_graph.py first: {PROCESSED / 'utas_research_graph.ttl'}")
    graph = Graph().parse(PROCESSED / "utas_research_graph.ttl", format="turtle")
    corpus = load_corpus(PROCESSED)
    semantic = SemanticRetriever(corpus, PROCESSED)
    retriever = HybridRetriever(corpus, None, semantic)
    project_rows = json.loads((PROCESSED / "project_documents.json").read_text(encoding="utf-8"))
    projects = [ProjectDocument.model_validate(row) for row in project_rows]

    evaluations = []

    # A: general UTAS guidance is present in retrieval, not the project-only graph.
    for question, query in zip(QUESTIONS[:3], [
        "What English score is required for a PhD?",
        "What documents are required when applying?",
        "What scholarships are available?",
    ]):
        results = _retrieval_search(retriever, query, scope="general")
        evaluations.append({
            "question": question[0], "query_type": "A", "recommended_method": "retrieval",
            "retrieval_result_summary": {"top_results": _titles(results),
                                         "note": "Hybrid retrieval surfaces relevant general guidance; it does not itself synthesize a final answer."},
            "graph_result_summary": {"result": "No general-degree guidance is represented in this project graph."},
            "retrieval_capable": bool(results), "graph_capable": False, "agreement": None,
            "graph_contribution": "None for these general-information questions; use the general-document corpus.",
        })

    # B: explicit metadata filters make the retrieval candidate set exhaustive and comparable.
    for question, filters in ((QUESTIONS[3][0], FILTERS_4), (QUESTIONS[4][0], FILTERS_5)):
        matching_projects = filter_projects(projects, **filters)
        retrieval = _retrieval_search(retriever, question, scope="projects", filters=filters, top_k=len(projects))
        graph_filters = {"degree_type": filters.get("degree_type"), "student_type": filters.get("student_type"),
                         "location": filters.get("location"), "funding_status": filters.get("funding_status"),
                         "application_status": filters.get("status"),
                         "research_category": filters.get("research_category")}
        graph_results = get_projects_by_constraints(graph, **graph_filters)
        retrieval_ids = {project.project_id for project in matching_projects}
        ranked_ids = {row["project_id"] for row in retrieval if row["item_type"] == "research_project"}
        graph_ids = {row["project_id"] for row in graph_results}
        agreement = compare_sets(ranked_ids, graph_ids)
        agreement["structured_filter_matches_graph"] = retrieval_ids == graph_ids
        agreement["hybrid_returned_all_eligible_ids"] = ranked_ids == retrieval_ids
        evaluations.append({
            "question": question, "query_type": "B", "recommended_method": "hybrid_graph",
            "retrieval_result_summary": {"matching_project_count": len(retrieval_ids),
                                         "top_results": _titles(retrieval),
                                         "hybrid_returned_all_eligible_ids": ranked_ids == retrieval_ids},
            "graph_result_summary": {"matching_project_count": len(graph_ids), "sample_project_ids": sorted(graph_ids)[:10],
                                     "sample_titles": _titles(graph_results)},
            "retrieval_capable": True, "graph_capable": True, "agreement": agreement,
            "graph_contribution": "Independent SPARQL verification of the complete conjunctive project set.",
        })

    # Exact single-project lookup: compare supervisor values directly.
    question = QUESTIONS[5][0]
    retrieval = _retrieval_search(retriever, question, scope="projects", filters={"project_id": "12259"}, top_k=10)
    graph_project = get_project_by_id(graph, "12259")
    retrieval_project = next((row for row in retrieval if row.get("project_id") == "12259"), None)
    left_supervisors = [retrieval_project["primary_supervisor"]] if retrieval_project and retrieval_project.get("primary_supervisor") else []
    right_supervisors = [graph_project["primary_supervisor"]] if graph_project and graph_project.get("primary_supervisor") else []
    supervisor_agreement = compare_sets(left_supervisors, right_supervisors)
    project_id_match = bool(retrieval_project and graph_project
                            and retrieval_project["project_id"] == graph_project["project_id"])
    evaluations.append({
        "question": question, "query_type": "B", "recommended_method": "hybrid_graph",
        "retrieval_result_summary": {"project_id": retrieval_project.get("project_id") if retrieval_project else None,
                                     "supervisor": left_supervisors[0] if left_supervisors else None},
        "graph_result_summary": {"project_id": graph_project.get("project_id") if graph_project else None,
                                 "title": graph_project.get("title") if graph_project else None,
                                 "supervisor": right_supervisors[0] if right_supervisors else None},
        "retrieval_capable": retrieval_project is not None, "graph_capable": graph_project is not None,
        "agreement": supervisor_agreement | {
            "match": supervisor_agreement["match"] and project_id_match,
            "project_id_match": project_id_match,
        },
        "graph_contribution": "Exact project-ID lookup and independent supervisor verification.",
    })

    # C: retrieve illustrative evidence, but do not mistake ranked snippets for exhaustive aggregates.
    aggregation_specs = [
        (QUESTIONS[6][0], supervisors_with_multiple_projects(graph),
         lambda rows: {"entity_count": len(rows), "sample": rows[:5]},
         "A ranked document list cannot reliably establish every supervisor's project count."),
        (QUESTIONS[7][0], supervisors_with_funded_ict_international_projects(graph),
         lambda rows: {"supervisor_count": len(rows), "sample": rows[:5]},
         "SPARQL returns the complete distinct supervisor set and matching project counts."),
        (QUESTIONS[8][0], count_open_projects_by_category(graph),
         lambda rows: {"category_count": len(rows), "counts_by_category": rows},
         "SPARQL groups the full graph and counts distinct open projects per category."),
        (QUESTIONS[9][0], categories_with_phd_and_masters_by_research(graph),
         lambda rows: {"category_count": len(rows), "categories": rows},
         "SPARQL checks both degree types within each category across all projects."),
        (QUESTIONS[10][0], supervisors_across_multiple_categories(graph),
         lambda rows: {"supervisor_count": len(rows), "sample": rows[:5]},
         "SPARQL groups categories by supervisor and returns only supervisors spanning multiple categories."),
    ]
    for question, graph_rows, summarize, contribution in aggregation_specs:
        retrieval = _retrieval_search(retriever, question, scope="projects", top_k=5)
        evaluations.append({
            "question": question, "query_type": "C", "recommended_method": "graph",
            "retrieval_result_summary": {"top_results": _titles(retrieval),
                                         "exact_aggregate": None,
                                         "note": "Retrieval may surface relevant project evidence but does not provide an exhaustive relationship/aggregation answer."},
            "graph_result_summary": summarize(graph_rows), "retrieval_capable": False,
            "graph_capable": True, "agreement": None,
            "graph_contribution": contribution,
        })

    # C12 is both filterable through retrieval metadata and independently countable in SPARQL.
    question = QUESTIONS[11][0]
    matching_projects = filter_projects(projects, **FILTERS_12)
    retrieval = _retrieval_search(retriever, question, scope="projects", filters=FILTERS_12, top_k=len(projects))
    graph_matches = get_projects_by_constraints(
        graph, degree_type="PhD", student_type="International", location=None,
        funding_status="funded", application_status="Applications open",
        research_category="Information and Communication Technology",
    )
    retrieval_ids = {project.project_id for project in matching_projects}
    graph_ids = {row["project_id"] for row in graph_matches}
    count_agreement = compare_counts(len(retrieval_ids), len(graph_ids))
    id_agreement = compare_sets({row["project_id"] for row in retrieval}, graph_ids)
    evaluations.append({
        "question": question, "query_type": "C", "recommended_method": "hybrid_graph",
        "retrieval_result_summary": {"matching_project_count": len(retrieval_ids), "top_results": _titles(retrieval),
                                     "hybrid_returned_all_eligible_ids": {row["project_id"] for row in retrieval} == retrieval_ids},
        "graph_result_summary": {"matching_project_count": len(graph_ids), "sample_project_ids": sorted(graph_ids)[:10]},
        "retrieval_capable": True, "graph_capable": True,
        "agreement": {
            "comparison_type": "count_and_set",
            "match": count_agreement["match"] and id_agreement["match"],
            "retrieval_count": count_agreement["left_count"], "graph_count": count_agreement["right_count"],
            "project_id_set_match": id_agreement["match"],
            "only_retrieval_ids": id_agreement["only_left"], "only_graph_ids": id_agreement["only_right"],
        },
        "graph_contribution": "SPARQL provides an independent exact count and matching project set for the same conjunction.",
    })

    return {"evaluation_method": "deterministic; no LLM judgment",
            "retrieval_corpus_items": corpus.summary,
            "evaluation_summary": summarize_evaluations(evaluations),
            "measurable_graph_benefits": [
                "Exact project ID and supervisor verification for project 12259.",
                "Exact, cross-checked results for conjunctive structured filters.",
                "Exhaustive group/count answers for supervisor and category relationships.",
            ],
            "evaluations": evaluations}


def print_table(report: dict) -> None:
    headers = ("Question", "Recommended method", "Retrieval capable?", "Graph capable?", "Agreement", "Graph contribution")
    rows = []
    for item in report["evaluations"]:
        agreement = item["agreement"]
        agreement_text = "n/a" if agreement is None else ("pass" if agreement["match"] else "fail")
        contribution = item["graph_contribution"]
        rows.append((item["question"], item["recommended_method"], str(item["retrieval_capable"]),
                     str(item["graph_capable"]), agreement_text, contribution))
    limits = (62, 18, 18, 14, 10, 48)
    widths = [min(limits[i], max(len(headers[i]), *(len(str(row[i])) for row in rows))) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        cells = [str(value) if len(str(value)) <= widths[i] else str(value)[:widths[i] - 1] + "…"
                 for i, value in enumerate(row)]
        print(" | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))))


def main() -> int:
    try:
        report = build_evaluation()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_table(report)
    print("\nEvaluation summary:")
    for key, value in report["evaluation_summary"].items():
        print(f"  {key}: {value}")
    print("Measurable graph benefits:")
    for benefit in report["measurable_graph_benefits"]:
        print(f"  - {benefit}")
    print(f"\nSaved: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
