"""Run representative SPARQL graph queries and structured-filter comparisons."""

import json
from pathlib import Path

from rdflib import Graph

from utas_research_assistant.graph.builder import graph_statistics
from utas_research_assistant.graph.queries import (
    categories_with_phd_and_masters_by_research,
    count_open_projects_by_category,
    get_projects_by_constraints,
    supervisors_across_multiple_categories,
    supervisors_with_funded_ict_international_projects,
    supervisors_with_multiple_projects,
)
from utas_research_assistant.retrieval.filters import filter_projects
from utas_research_assistant.retrieval.project_documents import ProjectDocument

ROOT = Path(__file__).resolve().parents[1]
GRAPH_FILE = ROOT / "data/processed/utas_research_graph.ttl"
PROJECTS_FILE = ROOT / "data/processed/project_documents.json"


def main() -> None:
    if not GRAPH_FILE.is_file():
        raise SystemExit(f"Graph file not found; run scripts/build_graph.py first: {GRAPH_FILE}")
    graph = Graph().parse(GRAPH_FILE, format="turtle")
    docs = [ProjectDocument.model_validate(item) for item in json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))]
    print("Graph statistics:")
    for key, value in graph_statistics(graph).items():
        print(f"  {key}: {value}")

    probes = [
        ("A. Supervisors with more than one advertised project", supervisors_with_multiple_projects(graph)),
        ("B. Supervisors with funded ICT projects accepting international students",
         supervisors_with_funded_ict_international_projects(graph)),
        ("C. Open projects by research category", count_open_projects_by_category(graph)),
        ("D. Categories with both PhD and Master by Research projects",
         categories_with_phd_and_masters_by_research(graph)),
        ("E. Supervisors across more than one research category",
         supervisors_across_multiple_categories(graph)),
    ]
    for title, rows in probes:
        print(f"\n{title} ({len(rows)} results)")
        for row in rows[:10]:
            print(f"  {row}")
        if len(rows) > 10:
            print("  ...")

    cases = [
        ("International + ICT + PhD + funded + open", {
            "degree_type": "PhD", "student_type": "International",
            "research_category": "Information and Communication Technology",
            "funding_status": "funded", "application_status": "Applications open",
        }, {"degree_type": "PhD", "student_type": "International",
            "research_category": "Information and Communication Technology",
            "funding_status": "funded", "status": "Applications open"}, 17),
        ("Master by Research + Hobart + open", {
            "degree_type": "Master by Research", "location": "Hobart",
            "application_status": "Applications open",
        }, {"degree_type": "Master by Research", "location": "Hobart",
            "status": "Applications open"}, 15),
    ]
    print("\nStructured-data cross-validation:")
    for name, graph_filters, structured_filters, expected in cases:
        graph_matches = get_projects_by_constraints(graph, **graph_filters)
        structured_matches = filter_projects(docs, **structured_filters)
        graph_count, structured_count = len(graph_matches), len(structured_matches)
        print(f"  {name}: SPARQL={graph_count}, structured={structured_count}, "
              f"match={graph_count == structured_count}, known baseline matches={structured_count == expected}")
        for project in graph_matches[:3]:
            print(f"    {project['project_id']}: {project['title']}")


if __name__ == "__main__":
    main()
