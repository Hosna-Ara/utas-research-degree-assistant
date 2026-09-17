"""Build a deterministic RDF graph from validated project documents."""

import hashlib
import re
from collections.abc import Iterable, Mapping
from urllib.parse import quote

from rdflib import Graph, Literal, RDF, RDFS, URIRef, XSD

from utas_research_assistant.graph.namespaces import UTAS
from utas_research_assistant.retrieval.project_documents import ProjectDocument
from utas_research_assistant.retrieval.supervisor_documents import SupervisorDocument

ENTITY_TYPES = {
    "supervisor": (UTAS.Supervisor, UTAS.supervisedBy),
    "research_category": (UTAS.ResearchCategory, UTAS.hasResearchCategory),
    "degree_type": (UTAS.DegreeType, UTAS.hasDegreeType),
    "student_type": (UTAS.StudentType, UTAS.acceptsStudentType),
    "location": (UTAS.Location, UTAS.locatedAt),
    "funding_status": (UTAS.FundingStatus, UTAS.hasFundingStatus),
    "application_status": (UTAS.ApplicationStatus, UTAS.hasApplicationStatus),
    "research_field": (UTAS.ResearchField, UTAS.hasResearchField),
    "school": (UTAS.School, UTAS.affiliatedWith),
}


def _clean(value: str) -> str:
    return " ".join(value.split())


def _key(value: str) -> str:
    return _clean(value).casefold()


def project_uri(project_id: str) -> URIRef:
    """Return a stable, URL-safe project URI keyed by the source project ID."""
    return UTAS["project/" + quote(_clean(str(project_id)), safe="")]


def entity_uri(entity_type: str, label: str) -> URIRef:
    """Return a stable named-entity URI; equal case/space variants share a node."""
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"Unknown entity type: {entity_type}")
    normalized = _key(label)
    if not normalized:
        raise ValueError("Entity labels must not be blank")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-") or "entity"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return UTAS[f"{entity_type}/{slug}-{digest}"]


def _document(record: ProjectDocument | Mapping) -> ProjectDocument:
    return record if isinstance(record, ProjectDocument) else ProjectDocument.model_validate(record)


def _add_entity(graph: Graph, project: URIRef, entity_type: str, label: str) -> None:
    label = _clean(label)
    if not label:
        return
    rdf_type, relationship = ENTITY_TYPES[entity_type]
    entity = entity_uri(entity_type, label)
    graph.add((entity, RDF.type, rdf_type))
    if not any(graph.objects(entity, RDFS.label)):
        graph.add((entity, RDFS.label, Literal(label, lang="en")))
    graph.add((project, relationship, entity))


def _supervisor_key(value: str) -> str:
    value = _clean(value).casefold()
    value = re.sub(r"^(?:associate\s+professor|assoc\.?\s+prof|professor|prof\.?|doctor|dr\.?)\s+", "", value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _supervisor_entity(graph: Graph, label: str, known: dict[str, URIRef]) -> URIRef:
    key = _supervisor_key(label)
    if key in known:
        return known[key]
    entity = entity_uri("supervisor", label)
    known[key] = entity
    graph.add((entity, RDF.type, UTAS.Supervisor))
    graph.add((entity, RDFS.label, Literal(_clean(label), lang="en")))
    return entity


def build_graph(records: Iterable[ProjectDocument | Mapping], supervisor_profiles: Iterable[SupervisorDocument | Mapping] | None = None) -> Graph:
    graph = Graph()
    graph.bind("utas", UTAS)
    graph.bind("rdfs", RDFS)
    graph.bind("xsd", XSD)
    supervisor_entities: dict[str, URIRef] = {}
    reconcile_supervisors = supervisor_profiles is not None
    for raw in records:
        record = _document(raw)
        project = project_uri(record.project_id)
        graph.add((project, RDF.type, UTAS.ResearchProject))
        graph.add((project, UTAS.projectId, Literal(record.project_id, datatype=XSD.string)))
        graph.add((project, UTAS.title, Literal(_clean(record.title), datatype=XSD.string)))
        for predicate, value in (
            (UTAS.description, record.description),
            (UTAS.scholarshipText, record.scholarship_text),
            (UTAS.closingDate, record.closing_date),
        ):
            if value and _clean(value):
                graph.add((project, predicate, Literal(_clean(value), datatype=XSD.string)))
        if record.source_url:
            graph.add((project, UTAS.sourceUrl, Literal(str(record.source_url), datatype=XSD.anyURI)))
        if record.primary_supervisor:
            if reconcile_supervisors:
                entity = _supervisor_entity(graph, record.primary_supervisor, supervisor_entities)
            else:
                _add_entity(graph, project, "supervisor", record.primary_supervisor)
                entity = entity_uri("supervisor", record.primary_supervisor)
            graph.add((project, UTAS.supervisedBy, entity))
            graph.add((entity, UTAS.supervises, project))
        for label in record.research_categories:
            _add_entity(graph, project, "research_category", label)
        for label in record.degree_types:
            _add_entity(graph, project, "degree_type", label)
        for label in record.student_types:
            _add_entity(graph, project, "student_type", label)
        if record.location:
            # Source values may list several campuses in one semicolon-delimited field.
            for label in record.location.split(";"):
                _add_entity(graph, project, "location", label)
        if record.funding_status:
            _add_entity(graph, project, "funding_status", record.funding_status)
        if record.status:
            _add_entity(graph, project, "application_status", record.status)
    for raw in supervisor_profiles or []:
        profile_data = raw.model_dump(mode="json") if isinstance(raw, SupervisorDocument) else dict(raw)
        canonical_name = profile_data.get("canonical_name") or profile_data.get("title") or ""
        if not canonical_name:
            continue
        entity = _supervisor_entity(graph, canonical_name, supervisor_entities)
        school = profile_data.get("school")
        research_fields = profile_data.get("research_fields") or []
        related_project_ids = profile_data.get("related_project_ids") or []
        for predicate, value in ((UTAS.canonicalName, canonical_name),
                                 (UTAS.title, profile_data.get("title")),
                                 (UTAS.school, school),
                                 (UTAS.discoveryProfileId, profile_data.get("discovery_profile_id")),
                                 (UTAS.discoveryProfileUrl, profile_data.get("discovery_url") or profile_data.get("source_url")),
                                 (UTAS.researchFieldsText, "; ".join(research_fields))):
            if value:
                graph.add((entity, predicate, Literal(_clean(str(value)), datatype=XSD.string)))
        for field in research_fields:
            _add_entity(graph, entity, "research_field", field)
        if school:
            _add_entity(graph, entity, "school", school)
        # Bio/profile fields are read from normalized supervisor data, never raw JSON.
        for predicate, key in ((UTAS.bio, "bio"), (UTAS.orcid, "orcid"), (UTAS.email, "email"),
                               (UTAS.updatedWhen, "updated_when"), (UTAS.availability, "availability")):
            value = profile_data.get(key)
            if value:
                graph.add((entity, predicate, Literal(_clean(str(value)), datatype=XSD.string)))
        for project_id in related_project_ids:
            project = project_uri(project_id)
            if (project, RDF.type, UTAS.ResearchProject) in graph:
                graph.add((entity, UTAS.supervises, project))
    return graph


def graph_statistics(graph: Graph) -> dict[str, int]:
    stats = {"total_triples": len(graph)}
    for key, rdf_type in (
        ("number_of_projects", UTAS.ResearchProject),
        ("number_of_supervisors", UTAS.Supervisor),
        ("number_of_research_categories", UTAS.ResearchCategory),
        ("number_of_degree_types", UTAS.DegreeType),
        ("number_of_student_types", UTAS.StudentType),
        ("number_of_locations", UTAS.Location),
        ("number_of_funding_status_nodes", UTAS.FundingStatus),
        ("number_of_application_status_nodes", UTAS.ApplicationStatus),
        ("number_of_research_fields", UTAS.ResearchField),
        ("number_of_schools", UTAS.School),
    ):
        stats[key] = sum(1 for _ in graph.subjects(RDF.type, rdf_type))
    return stats
