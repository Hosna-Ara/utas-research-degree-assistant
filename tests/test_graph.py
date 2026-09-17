import json
import pytest

from rdflib import Graph, RDF, RDFS, Literal, XSD

from utas_research_assistant.graph.builder import build_graph, entity_uri, graph_statistics, project_uri
from utas_research_assistant.graph.namespaces import UTAS
from utas_research_assistant.graph.queries import (
    categories_with_phd_and_masters_by_research,
    count_open_funded_international_ict_projects,
    count_open_projects_by_category,
    get_project_by_id,
    get_projects_by_constraints,
    get_projects_by_supervisor,
    supervisors_across_multiple_categories,
    supervisors_with_multiple_projects,
    SupervisorAmbiguityError, canonical_supervisor_name, resolve_supervisor_names,
    get_ict_supervisors_by_research_field,
)
from utas_research_assistant.retrieval.filters import filter_projects
from utas_research_assistant.retrieval.project_documents import ProjectDocument


def doc(project_id="1", **overrides):
    values = {
        "document_id": f"project-{project_id}", "project_id": project_id,
        "text": "Title: Project", "title": f"Project {project_id}",
        "description": "Project description", "source_url": f"https://example.org/project/{project_id}",
        "status": "Applications open", "degree_types": ["PhD"],
        "student_types": ["International"], "location": "Hobart",
        "scholarship_text": "$34,315 pa", "funding_status": "funded",
        "closing_date": "30 November 2026", "primary_supervisor": "  Dr   Alex Smith ",
        "research_categories": ["Information and Communication Technology"],
    }
    values.update(overrides)
    return ProjectDocument.model_validate(values)


def fixture_graph():
    records = [
        doc("1", research_categories=["Information and Communication Technology", "Data Science"]),
        doc("2", primary_supervisor="dr alex smith", degree_types=["Master by Research"],
            research_categories=["Information and Communication Technology"]),
        doc("3", primary_supervisor="Dr Taylor", student_types=["Domestic"],
            funding_status="unknown", scholarship_text=None, status="Under assessment",
            research_categories=["Data Science"]),
    ]
    return records, build_graph(records)


def test_deterministic_safe_uri_creation_and_conservative_entity_normalization():
    assert project_uri("12167") == project_uri("12167")
    assert "%2F" in str(project_uri("x/y"))
    assert entity_uri("supervisor", " Dr   Alex Smith ") == entity_uri("supervisor", "dr alex smith")
    assert entity_uri("supervisor", "Alex Smith") != entity_uri("supervisor", "Alex Smyth")


def test_project_node_properties_and_relationships():
    records, graph = fixture_graph()
    project = project_uri("1")
    assert (project, RDF.type, UTAS.ResearchProject) in graph
    assert (project, UTAS.title, Literal(records[0].title, datatype=XSD.string)) in graph
    assert len(list(graph.objects(project, UTAS.supervisedBy))) == 1
    assert len(list(graph.objects(project, UTAS.hasResearchCategory))) == 2
    assert list(graph.objects(project, UTAS.hasFundingStatus))


def test_supervisor_deduplication_and_supervisor_lookup():
    _, graph = fixture_graph()
    assert graph_statistics(graph)["number_of_supervisors"] == 2
    assert len(get_projects_by_supervisor(graph, "DR ALEX SMITH")) == 2
    assert len(get_projects_by_supervisor(graph, "alex smith")) == 2
    assert len(get_projects_by_supervisor(graph, "  Doctor   Alex Smith  ")) == 2


def test_supervisor_title_normalization_and_ambiguity():
    records = [doc("1", primary_supervisor="Professor Alex Smith"),
               doc("2", primary_supervisor="Prof Alex Smith")]
    graph = build_graph(records)
    assert canonical_supervisor_name("Associate Professor, Alex Smith") == "alex smith"
    assert resolve_supervisor_names(graph, "Dr Alex Smith") == ["Prof Alex Smith", "Professor Alex Smith"]
    with pytest.raises(SupervisorAmbiguityError):
        get_projects_by_supervisor(graph, "Alex Smith")
    assert get_projects_by_supervisor(graph, "Unknown Person") == []


def test_serialization_reload_and_project_lookup(tmp_path):
    _, graph = fixture_graph()
    target = tmp_path / "graph.ttl"
    graph.serialize(target, format="turtle")
    loaded = Graph().parse(target, format="turtle")
    found = get_project_by_id(loaded, "2")
    assert found["title"] == "Project 2"
    assert get_project_by_id(loaded, "missing") is None


def test_multi_constraint_sparql_and_structured_data_consistency():
    records, graph = fixture_graph()
    constraints = {"degree_type": "PhD", "student_type": "International", "location": "Hobart",
                   "funding_status": "funded", "application_status": "open",
                   "research_category": "ICT"}
    graph_results = get_projects_by_constraints(graph, **constraints)
    structured = filter_projects(records, degree_type="PhD", student_type="International", location="Hobart",
                                funding_status="funded", status="open", research_category="ICT")
    assert [r["project_id"] for r in graph_results] == [r.project_id for r in structured] == ["1"]


def test_aggregation_queries_are_sparql_backed_and_handle_empty_results():
    _, graph = fixture_graph()
    assert {row["supervisor"] for row in supervisors_with_multiple_projects(graph)} == {"Dr Alex Smith"}
    assert "Dr Alex Smith" in {row["supervisor"] for row in supervisors_across_multiple_categories(graph)}
    assert count_open_projects_by_category(graph)
    assert "Information and Communication Technology" in categories_with_phd_and_masters_by_research(graph)
    empty = build_graph([])
    assert get_projects_by_constraints(empty, location="Moon") == []
    assert supervisors_with_multiple_projects(empty) == []
    assert count_open_projects_by_category(empty) == []
    assert count_open_funded_international_ict_projects(empty) == 0


def test_count_open_funded_international_ict_projects_uses_all_constraints():
    _, graph = fixture_graph()
    assert count_open_funded_international_ict_projects(graph) == 1
    closed_graph = build_graph([doc("9", status="Under assessment")])
    assert count_open_funded_international_ict_projects(closed_graph) == 0


def test_ict_field_query_intersects_project_category_not_profile_field_only():
    records = [
        doc("ict", primary_supervisor="Doctor ICT AI", research_categories=["Information and Communication Technology"]),
        doc("other", primary_supervisor="Doctor Other AI", research_categories=["Data Science"]),
    ]
    graph = build_graph(records)
    # Add source-derived AI fields to both profiles; only the ICT project-linked
    # supervisor is eligible for the constrained query.
    for name in ("Doctor ICT AI", "Doctor Other AI"):
        node = next(node for node in graph.subjects(RDF.type, UTAS.Supervisor)
                    if str(next(graph.objects(node, RDFS.label))) == name)
        field = entity_uri("research_field", "Artificial intelligence")
        graph.add((node, UTAS.hasResearchField, field))
        graph.add((field, RDFS.label, Literal("Artificial intelligence")))
    result = get_ict_supervisors_by_research_field(graph, "Artificial intelligence")
    assert [row["supervisor"] for row in result] == ["Doctor ICT AI"]
