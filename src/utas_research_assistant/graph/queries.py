"""Reusable local SPARQL queries over the UTAS research-project graph."""

import re

from rdflib import Graph, Literal, RDF, RDFS

from utas_research_assistant.graph.namespaces import UTAS
from utas_research_assistant.retrieval.filters import canonical

BASE = """
PREFIX utas: <https://example.org/utas-research/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

BASE_SELECT = """
SELECT DISTINCT ?project ?projectId ?title ?sourceUrl ?description ?scholarshipText ?closingDate
WHERE {
  ?project rdf:type utas:ResearchProject ; utas:projectId ?projectId ; utas:title ?title .
  OPTIONAL { ?project utas:sourceUrl ?sourceUrl }
  OPTIONAL { ?project utas:description ?description }
  OPTIONAL { ?project utas:scholarshipText ?scholarshipText }
  OPTIONAL { ?project utas:closingDate ?closingDate }
  %s
}
ORDER BY ?projectId
"""

RELATIONS = {
    "degree_type": ("hasDegreeType", "DegreeType"),
    "student_type": ("acceptsStudentType", "StudentType"),
    "location": ("locatedAt", "Location"),
    "funding_status": ("hasFundingStatus", "FundingStatus"),
    "status": ("hasApplicationStatus", "ApplicationStatus"),
    "research_category": ("hasResearchCategory", "ResearchCategory"),
    "supervisor": ("supervisedBy", "Supervisor"),
}


def _label(value) -> str:
    return str(value) if value is not None else ""


def canonical_supervisor_name(value: str) -> str:
    """Conservatively normalize names while removing only academic honorifics."""
    normalized = " ".join(str(value).strip().split()).casefold()
    normalized = re.sub(r"^[,\.\s]+|[,\.\s]+$", "", normalized)
    honorifics = (
        r"associate\s+professor", r"assoc\.?\s+prof", r"professor", r"prof\.?",
        r"doctor", r"dr\.?",
    )
    changed = True
    while changed:
        changed = False
        for honorific in honorifics:
            stripped = re.sub(rf"^{honorific}[\s,\.:-]+", "", normalized).strip()
            if stripped != normalized:
                normalized, changed = stripped, True
                break
    return re.sub(r"[^\w\s'-]", "", normalized, flags=re.UNICODE)


def resolve_supervisor_names(graph: Graph, supervisor_name: str) -> list[str]:
    """Resolve a user name to stored labels without fuzzy matching."""
    wanted = canonical_supervisor_name(supervisor_name)
    labels = sorted({str(label) for node in graph.subjects(RDF.type, UTAS.Supervisor)
                     for label in graph.objects(node, RDFS.label)})
    return [label for label in labels if canonical_supervisor_name(label) == wanted]


def _supervisor_nodes(graph: Graph, supervisor_name: str) -> list:
    matches = resolve_supervisor_names(graph, supervisor_name)
    return [node for node in graph.subjects(RDF.type, UTAS.Supervisor)
            if any(str(label) in matches for label in graph.objects(node, RDFS.label))]


def get_supervisor_profile(graph: Graph, supervisor_name: str) -> dict | None:
    matches = resolve_supervisor_names(graph, supervisor_name)
    if len(matches) > 1:
        raise SupervisorAmbiguityError(supervisor_name, matches)
    nodes = _supervisor_nodes(graph, supervisor_name)
    if not nodes:
        return None
    node = nodes[0]
    def literal(predicate):
        value = next(iter(graph.objects(node, predicate)), None)
        return str(value) if value is not None else None
    fields = sorted({str(label) for field in graph.objects(node, UTAS.hasResearchField)
                     for label in graph.objects(field, RDFS.label)})
    projects = sorted({str(graph.value(project, UTAS.projectId)) for project in graph.subjects(UTAS.supervisedBy, node)
                       if graph.value(project, UTAS.projectId) is not None})
    return {
        "canonical_name": literal(UTAS.canonicalName) or matches[0],
        "title": literal(UTAS.title), "school": literal(UTAS.school),
        "bio": literal(UTAS.bio), "orcid": literal(UTAS.orcid),
        "email": literal(UTAS.email), "discovery_url": literal(UTAS.discoveryProfileUrl),
        "source_url": literal(UTAS.discoveryProfileUrl),
        "research_fields": fields, "related_project_ids": projects,
    }


def get_supervisor_research_fields(graph: Graph, supervisor_name: str) -> dict | None:
    profile = get_supervisor_profile(graph, supervisor_name)
    if profile is None:
        return None
    return {"canonical_name": profile["canonical_name"], "research_fields": profile["research_fields"],
            "source_url": profile.get("discovery_url")}


def get_supervisors_by_research_field(graph: Graph, field: str) -> list[dict]:
    query = BASE + """
SELECT DISTINCT ?supervisor ?name ?school WHERE {
 ?supervisor a utas:Supervisor ; rdfs:label ?name ; utas:hasResearchField ?field .
 ?field rdfs:label ?fieldLabel . OPTIONAL { ?supervisor utas:school ?school }
 FILTER(LCASE(STR(?fieldLabel)) = LCASE(?wanted))
}
ORDER BY LCASE(STR(?name))
"""
    return [{"supervisor": str(row.name), "school": _label(row.school) or None}
            for row in graph.query(query, initBindings={"wanted": Literal(field)})]


def get_ict_supervisors_by_research_field(graph: Graph, field: str) -> list[dict]:
    """Return AI/field supervisors intersected with advertised ICT projects."""
    query = BASE + """
SELECT DISTINCT ?name ?school WHERE {
 ?project a utas:ResearchProject ; utas:supervisedBy ?supervisor ; utas:hasResearchCategory ?category .
 ?category rdfs:label ?categoryLabel .
 ?supervisor a utas:Supervisor ; rdfs:label ?name ; utas:hasResearchField ?fieldNode .
 ?fieldNode rdfs:label ?fieldLabel .
 OPTIONAL { ?supervisor utas:school ?school }
 FILTER(LCASE(STR(?categoryLabel)) = "information and communication technology")
 FILTER(LCASE(STR(?fieldLabel)) = LCASE(?wanted))
}
ORDER BY LCASE(STR(?name))
"""
    return [{"supervisor": str(row.name), "school": _label(row.school) or None}
            for row in graph.query(query, initBindings={"wanted": Literal(field)})]


def get_supervisors_by_school(graph: Graph, school: str) -> list[dict]:
    query = BASE + """
SELECT DISTINCT ?name ?school WHERE {
 ?supervisor a utas:Supervisor ; rdfs:label ?name ; utas:school ?school .
 FILTER(LCASE(STR(?school)) = LCASE(?wanted))
} ORDER BY LCASE(STR(?name))
"""
    return [{"supervisor": str(row.name), "school": str(row.school)}
            for row in graph.query(query, initBindings={"wanted": Literal(school)})]


def get_ict_supervisors(graph: Graph) -> list[dict]:
    query = BASE + """
SELECT DISTINCT ?name ?school WHERE {
 ?project a utas:ResearchProject ; utas:supervisedBy ?supervisor ; utas:hasResearchCategory ?category .
 ?category rdfs:label ?categoryLabel . ?supervisor rdfs:label ?name .
 OPTIONAL { ?supervisor utas:school ?school }
 FILTER(LCASE(STR(?categoryLabel)) = "information and communication technology")
} ORDER BY LCASE(STR(?name))
"""
    return [{"supervisor": str(row.name), "school": _label(row.school) or None}
            for row in graph.query(query)]


def get_supervisors_with_orcid(graph: Graph) -> list[dict]:
    query = BASE + """
SELECT DISTINCT ?name ?orcid WHERE {
 ?supervisor a utas:Supervisor ; rdfs:label ?name ; utas:orcid ?orcid .
 FILTER(STRLEN(STR(?orcid)) > 0)
} ORDER BY LCASE(STR(?name))
"""
    return [{"supervisor": str(row.name), "orcid": str(row.orcid)}
            for row in graph.query(query)]


def get_projects_supervised_by_research_field(graph: Graph, field: str, **constraints) -> list[dict]:
    projects = get_projects_by_constraints(graph, **constraints)
    output = []
    for project in projects:
        project_row = get_project_by_id(graph, project["project_id"])
        supervisor = project_row.get("primary_supervisor") if project_row else None
        if supervisor and any(canonical_supervisor_name(row["supervisor"]) == canonical_supervisor_name(supervisor)
                              for row in get_supervisors_by_research_field(graph, field)):
            output.append(project)
    return output


class SupervisorAmbiguityError(ValueError):
    """Raised when conservative normalization maps to multiple people."""

    def __init__(self, requested: str, matches: list[str]):
        self.matches = matches
        super().__init__(f"Supervisor name is ambiguous: {requested!r} matches {', '.join(matches)}")


def _project_data(graph: Graph, row) -> dict:
    project = row.project
    data = {
        "project_id": _label(row.projectId), "title": _label(row.title),
        "source_url": _label(row.sourceUrl), "description": _label(row.description) or None,
        "scholarship_text": _label(row.scholarshipText) or None,
        "closing_date": _label(row.closingDate) or None,
    }
    for key, (predicate, _) in RELATIONS.items():
        labels = sorted({_label(label) for node in graph.objects(project, getattr(UTAS, predicate))
                         for label in graph.objects(node, RDFS.label)})
        data[key if key not in {"degree_type", "student_type", "research_category"} else {
            "degree_type": "degree_types", "student_type": "student_types",
            "research_category": "research_categories",
        }[key]] = labels if key in {"degree_type", "student_type", "research_category"} else (labels[0] if labels else None)
    locations = sorted({_label(label) for node in graph.objects(project, UTAS.locatedAt)
                        for label in graph.objects(node, RDFS.label)})
    data["location"] = "; ".join(locations) if locations else None
    data["funding_status"] = data.pop("funding_status")
    data["status"] = data.pop("status")
    data["location"] = data.pop("location")
    data["primary_supervisor"] = data.pop("supervisor")
    if data["primary_supervisor"]:
        profile = get_supervisor_profile(graph, data["primary_supervisor"])
        if profile:
            data["supervisor_profile"] = profile
    return data


def _find_projects(graph: Graph, filters: dict[str, str | None] | None = None,
                   project_id: str | None = None) -> list[dict]:
    patterns = []
    bindings = {}
    if project_id is not None:
        patterns.append("FILTER(STR(?projectId) = ?wantedProjectId)")
        bindings["wantedProjectId"] = Literal(str(project_id))
    for index, (field, requested) in enumerate((filters or {}).items()):
        if requested is None:
            continue
        if field == "project_id":
            patterns.append(f"FILTER(STR(?projectId) = ?wanted{index})")
            bindings[f"wanted{index}"] = Literal(str(requested))
            continue
        if field not in RELATIONS:
            raise ValueError(f"Unknown graph filter: {field}")
        predicate, _ = RELATIONS[field]
        variable = f"entity{index}"
        label_variable = f"label{index}"
        patterns.append(f"?project utas:{predicate} ?{variable} . ?{variable} rdfs:label ?{label_variable} .")
        patterns.append(f"FILTER(LCASE(STR(?{label_variable})) = ?wanted{index})")
        bindings[f"wanted{index}"] = Literal(canonical(field, requested))
    query = BASE + BASE_SELECT % "\n  ".join(patterns)
    rows = graph.query(query, initBindings=bindings)
    return [_project_data(graph, row) for row in rows]


def get_project_by_id(graph: Graph, project_id: str) -> dict | None:
    matches = _find_projects(graph, project_id=project_id)
    return matches[0] if matches else None


def get_projects_by_supervisor(graph: Graph, supervisor_name: str) -> list[dict]:
    matches = resolve_supervisor_names(graph, supervisor_name)
    if len(matches) > 1:
        raise SupervisorAmbiguityError(supervisor_name, matches)
    if not matches:
        return []
    return _find_projects(graph, {"supervisor": matches[0]})


def get_projects_by_category(graph: Graph, category: str) -> list[dict]:
    return _find_projects(graph, {"research_category": category})


def get_projects_by_location(graph: Graph, location: str) -> list[dict]:
    return _find_projects(graph, {"location": location})


def get_projects_accepting_student_type(graph: Graph, student_type: str) -> list[dict]:
    return _find_projects(graph, {"student_type": student_type})


def get_funded_projects(graph: Graph) -> list[dict]:
    return _find_projects(graph, {"funding_status": "funded"})


def get_open_projects(graph: Graph) -> list[dict]:
    return _find_projects(graph, {"status": "Applications open"})


def get_projects_by_constraints(graph: Graph, *, degree_type=None, student_type=None, location=None,
                                funding_status=None, application_status=None,
                                research_category=None, supervisor=None, project_id=None) -> list[dict]:
    """SPARQL conjunction of every supplied structured project constraint."""
    return _find_projects(graph, {
        "degree_type": degree_type, "student_type": student_type, "location": location,
        "funding_status": funding_status, "status": application_status,
        "research_category": research_category, "supervisor": supervisor, "project_id": project_id,
    })


def count_open_funded_international_ict_projects(graph: Graph) -> int:
    query = BASE + """
SELECT (COUNT(DISTINCT ?project) AS ?project_count) WHERE {
 ?project a utas:ResearchProject ; utas:hasDegreeType ?d ; utas:acceptsStudentType ?t ;
   utas:hasFundingStatus ?f ; utas:hasApplicationStatus ?s ;
   utas:hasResearchCategory ?c .
 ?d rdfs:label ?degree . ?t rdfs:label ?student .
 ?f rdfs:label ?funding . ?s rdfs:label ?status . ?c rdfs:label ?category .
 FILTER(LCASE(STR(?degree)) = "phd" && LCASE(STR(?student)) = "international"
   && LCASE(STR(?funding)) = "funded" && LCASE(STR(?status)) = "applications open"
   && LCASE(STR(?category)) = "information and communication technology")
}
"""
    row = next(iter(graph.query(query)), None)
    return int(row.project_count) if row is not None else 0


def count_projects_by_category(graph: Graph) -> list[dict]:
    query = BASE + """
SELECT ?category (COUNT(DISTINCT ?project) AS ?project_count) WHERE {
  ?project a utas:ResearchProject ; utas:hasResearchCategory ?node . ?node rdfs:label ?category .
} GROUP BY ?category ORDER BY DESC(?project_count) LCASE(STR(?category))
"""
    return [{"category": str(r.category), "count": int(r.project_count)} for r in graph.query(query)]


def count_projects_by_supervisor(graph: Graph) -> list[dict]:
    query = BASE + """
SELECT ?supervisor (COUNT(DISTINCT ?project) AS ?project_count) WHERE {
  ?project a utas:ResearchProject ; utas:supervisedBy ?node . ?node rdfs:label ?supervisor .
} GROUP BY ?supervisor ORDER BY DESC(?project_count) LCASE(STR(?supervisor))
"""
    return [{"supervisor": str(r.supervisor), "count": int(r.project_count)} for r in graph.query(query)]


def supervisors_with_multiple_projects(graph: Graph) -> list[dict]:
    query = BASE + """
SELECT ?supervisor (COUNT(DISTINCT ?project) AS ?project_count) WHERE {
 ?project a utas:ResearchProject ; utas:supervisedBy ?node . ?node rdfs:label ?supervisor .
} GROUP BY ?node ?supervisor HAVING(COUNT(DISTINCT ?project) > 1)
ORDER BY DESC(?project_count) LCASE(STR(?supervisor))
"""
    return [{"supervisor": str(r.supervisor), "project_count": int(r.project_count)} for r in graph.query(query)]


def supervisors_with_funded_ict_international_projects(graph: Graph) -> list[dict]:
    query = BASE + """
SELECT ?supervisor (COUNT(DISTINCT ?project) AS ?project_count) WHERE {
 ?project a utas:ResearchProject ;
   utas:supervisedBy ?s ; utas:hasFundingStatus ?f ; utas:hasResearchCategory ?c ;
   utas:acceptsStudentType ?t .
 ?s rdfs:label ?supervisor . ?f rdfs:label ?fund . ?c rdfs:label ?category . ?t rdfs:label ?student .
 FILTER(LCASE(STR(?fund)) = "funded" && LCASE(STR(?student)) = "international"
   && LCASE(STR(?category)) = "information and communication technology")
} GROUP BY ?s ?supervisor ORDER BY LCASE(STR(?supervisor))
"""
    return [{"supervisor": str(r.supervisor), "project_count": int(r.project_count)}
            for r in graph.query(query)]


def count_open_projects_by_category(graph: Graph) -> list[dict]:
    query = BASE + """
SELECT ?category (COUNT(DISTINCT ?project) AS ?project_count) WHERE {
 ?project a utas:ResearchProject ; utas:hasApplicationStatus ?status ; utas:hasResearchCategory ?c .
 ?status rdfs:label ?statusLabel . ?c rdfs:label ?category .
 FILTER(LCASE(STR(?statusLabel)) = "applications open")
} GROUP BY ?category ORDER BY DESC(?project_count) LCASE(STR(?category))
"""
    return [{"category": str(r.category), "count": int(r.project_count)} for r in graph.query(query)]


def categories_with_phd_and_masters_by_research(graph: Graph) -> list[str]:
    query = BASE + """
SELECT DISTINCT ?category WHERE {
 ?phd a utas:ResearchProject ; utas:hasResearchCategory ?c ; utas:hasDegreeType ?d1 .
 ?c rdfs:label ?category . ?d1 rdfs:label ?phdLabel .
 FILTER(LCASE(STR(?phdLabel)) = "phd")
 ?masters a utas:ResearchProject ; utas:hasResearchCategory ?c ; utas:hasDegreeType ?d2 .
 ?d2 rdfs:label ?mastersLabel . FILTER(LCASE(STR(?mastersLabel)) = "master by research")
} ORDER BY LCASE(STR(?category))
"""
    return [str(r.category) for r in graph.query(query)]


def supervisors_across_multiple_categories(graph: Graph) -> list[dict]:
    query = BASE + """
SELECT ?supervisor (COUNT(DISTINCT ?category) AS ?category_count) WHERE {
 ?project a utas:ResearchProject ; utas:supervisedBy ?s ; utas:hasResearchCategory ?c .
 ?s rdfs:label ?supervisor . ?c rdfs:label ?category .
} GROUP BY ?s ?supervisor HAVING(COUNT(DISTINCT ?category) > 1)
ORDER BY DESC(?category_count) LCASE(STR(?supervisor))
"""
    return [{"supervisor": str(r.supervisor), "category_count": int(r.category_count)}
            for r in graph.query(query)]
