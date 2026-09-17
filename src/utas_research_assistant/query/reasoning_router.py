"""Route plans to hybrid retrieval, controlled SPARQL operations, or both."""

from dataclasses import dataclass
import re

from rdflib import Graph, RDF, RDFS
from utas_research_assistant.graph.namespaces import UTAS

from utas_research_assistant.graph.queries import (
    categories_with_phd_and_masters_by_research,
    count_open_funded_international_ict_projects,
    count_open_projects_by_category,
    get_project_by_id,
    get_projects_by_constraints,
    get_projects_by_supervisor,
    supervisors_across_multiple_categories,
    supervisors_with_funded_ict_international_projects,
    supervisors_with_multiple_projects,
    get_supervisor_profile, get_supervisor_research_fields, get_supervisors_by_research_field,
    get_ict_supervisors_by_research_field,
    get_ict_supervisors, get_supervisors_by_school, get_supervisors_with_orcid,
    get_projects_supervised_by_research_field,
    canonical_supervisor_name,
)
from utas_research_assistant.query.models import ReasoningPlan
from utas_research_assistant.query.planner import ReasoningPlanner
from utas_research_assistant.retrieval.hybrid import HybridRetriever

GRAPH_OPERATIONS = {
    "project_by_id", "projects_by_supervisor", "multi_constraint_projects",
    "supervisors_with_multiple_projects", "supervisors_with_funded_ict_international_projects",
    "open_projects_by_category", "categories_with_both_degree_types",
    "supervisors_across_multiple_categories", "count_open_funded_international_ict_projects",
    "supervisor_by_name", "supervisor_research_fields", "supervisors_by_research_field",
    "ict_supervisors", "ict_supervisors_by_research_field", "supervisor_school", "supervisors_with_orcid", "projects_by_supervisor_research_field",
}


def _project_constraints(plan: ReasoningPlan) -> dict:
    filters = plan.project_filters()
    return {key: filters[key] for key in (
        "degree_type", "student_type", "location", "funding_status", "status",
        "research_category", "supervisor", "project_id",
    ) if key in filters}


def execute_graph_operation(graph: Graph, plan: ReasoningPlan):
    """Dispatch only named, reviewed operations; no query text comes from the LLM."""
    operation = plan.graph_operation
    if operation not in GRAPH_OPERATIONS:
        raise ValueError(f"Unsupported graph operation: {operation!r}")
    filters = _project_constraints(plan)
    if operation == "project_by_id":
        return get_project_by_id(graph, plan.project_id or "")
    if operation == "projects_by_supervisor":
        if not plan.supervisor:
            return []
        return get_projects_by_supervisor(graph, plan.supervisor)
    if operation == "multi_constraint_projects":
        return get_projects_by_constraints(graph, **filters)
    if operation == "supervisors_with_multiple_projects":
        return supervisors_with_multiple_projects(graph)
    if operation == "supervisors_with_funded_ict_international_projects":
        return supervisors_with_funded_ict_international_projects(graph)
    if operation == "open_projects_by_category":
        return count_open_projects_by_category(graph)
    if operation == "categories_with_both_degree_types":
        return categories_with_phd_and_masters_by_research(graph)
    if operation == "supervisors_across_multiple_categories":
        return supervisors_across_multiple_categories(graph)
    if operation == "count_open_funded_international_ict_projects":
        matches = get_projects_by_constraints(
            graph, degree_type="PhD", student_type="International", funding_status="funded",
            application_status="Applications open", research_category="Information and Communication Technology",
        )
        return {"count": count_open_funded_international_ict_projects(graph), "projects": matches}
    if operation == "supervisor_by_name":
        return get_supervisor_profile(graph, plan.supervisor or "")
    if operation == "supervisor_research_fields":
        return get_supervisor_research_fields(graph, plan.supervisor or "")
    if operation == "supervisors_by_research_field":
        return get_supervisors_by_research_field(graph, plan.research_category or plan.search_query)
    if operation == "ict_supervisors_by_research_field":
        return get_ict_supervisors_by_research_field(graph, plan.research_category or plan.search_query)
    if operation == "ict_supervisors":
        return get_ict_supervisors(graph)
    if operation == "supervisor_school":
        profile = get_supervisor_profile(graph, plan.supervisor or "")
        return {"canonical_name": profile["canonical_name"], "school": profile.get("school"),
                "source_url": profile.get("discovery_url")} if profile else None
    if operation == "supervisors_with_orcid":
        return get_supervisors_with_orcid(graph)
    if operation == "projects_by_supervisor_research_field":
        field = "Artificial intelligence" if re.search(r"\b(?:ai|artificial intelligence)\b", plan.search_query, re.I) else plan.search_query
        return get_projects_supervised_by_research_field(
            graph, field,
            degree_type=plan.degree_type, student_type=plan.student_type,
            location=plan.location, funding_status=plan.funding_status,
            application_status=plan.status, research_category=plan.research_category,
        )
    raise ValueError(f"Unsupported graph operation: {operation!r}")


def _walk_values(value, key: str):
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                if isinstance(child, list):
                    yield from child
                elif child is not None:
                    yield child
            else:
                yield from _walk_values(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child, key)


def _source_metadata(retrieval_results: list[dict], graph_result) -> tuple[list[str], list[str]]:
    sources = list(_walk_values(retrieval_results, "source_url"))
    sources.extend(_walk_values(graph_result, "source_url"))
    ids = list(_walk_values(retrieval_results, "project_id"))
    ids.extend(_walk_values(graph_result, "project_id"))
    # Stable deduplication while preserving the first displayed occurrence.
    return list(dict.fromkeys(map(str, sources))), list(dict.fromkeys(map(str, ids)))


def _result_count(result) -> int:
    if result is None:
        return 0
    if isinstance(result, dict) and "count" in result:
        return int(result["count"])
    if isinstance(result, list):
        return len(result)
    return 1


def _exact_supervisor_project_plan(graph: Graph, question: str, plan: ReasoningPlan) -> ReasoningPlan:
    """Prefer exact full-name matching for supervisor project questions."""
    if not re.search(r"\b(?:projects?|supervis(?:e|es|ed|or)|under)\b", question, re.I):
        return plan
    normalized_question = " ".join(re.sub(r"[^\w\s'-]", " ", question.casefold()).split())
    candidates = []
    for node in graph.subjects(RDF.type, UTAS.Supervisor):
        for label in graph.objects(node, RDFS.label):
            canonical = canonical_supervisor_name(str(label))
            if len(canonical.split()) >= 2 and re.search(rf"(?<![\w]){re.escape(canonical)}(?![\w])", normalized_question):
                candidates.append(str(label))
    names = sorted(set(candidates))
    if len(names) != 1:
        return plan
    matched = names[0]
    return plan.model_copy(update={
        "method": "graph", "scope": "projects", "search_query": matched,
        "supervisor": matched, "graph_operation": "projects_by_supervisor",
        "intent": "supervisor_project_lookup", "confidence": max(plan.confidence or 0, 0.9),
    })


def _ambiguous_first_name(graph: Graph, question: str) -> list[str]:
    """Return matching people when a project query supplies only a shared first name."""
    if not re.search(r"\b(?:projects?|supervis(?:e|es|ed|or)|under)\b", question, re.I):
        return []
    words = set(re.findall(r"[a-z]+", question.casefold()))
    grouped: dict[str, set[str]] = {}
    for node in graph.subjects(RDF.type, UTAS.Supervisor):
        for label in graph.objects(node, RDFS.label):
            canonical = canonical_supervisor_name(str(label))
            parts = canonical.split()
            if len(parts) >= 2:
                grouped.setdefault(parts[0], set()).add(str(label))
    for first_name, labels in sorted(grouped.items()):
        if first_name in words and len(labels) > 1:
            return sorted(labels)
    return []


@dataclass
class ReasoningRouter:
    planner: ReasoningPlanner
    retriever: HybridRetriever
    graph: Graph
    top_k: int = 5

    def route(self, question: str) -> dict:
        plan = self.planner.plan(question)
        plan = _exact_supervisor_project_plan(self.graph, question, plan)
        # Preserve the explicit ICT-project universe even if a local model
        # selects the broader field operation.
        if (plan.graph_operation == "supervisors_by_research_field"
                and re.search(r"\bict\b", question, re.I)
                and re.search(r"\b(?:ai|artificial intelligence)\b", question, re.I)):
            plan = plan.model_copy(update={"graph_operation": "ict_supervisors_by_research_field"})
        ambiguous = _ambiguous_first_name(self.graph, question)
        if ambiguous and plan.graph_operation != "projects_by_supervisor":
            return {
                "original_question": question, "reasoning_method": "graph", "scope": "projects",
                "planner_method": plan.planner_method, "interpreted_intent": "ambiguous_supervisor_lookup",
                "extracted_constraints": {}, "graph_operation_used": None,
                "candidate_count_before": 0, "candidate_count_after": 0,
                "ranked_retrieval_evidence": [], "graph_result": None, "source_urls": [],
                "project_ids": [], "supervisor_ambiguity": ambiguous,
                "tool_trace": [{"step": "supervisor_name_disambiguation", "candidate_names": ambiguous}],
            }
        filters = _project_constraints(plan)
        retrieval_evidence = []
        graph_result = None
        operation_used = None
        before = len(self.retriever.candidate_indices(plan.scope))
        after = before
        trace = [{"step": "plan_question", "planner_method": plan.planner_method,
                  "reasoning_method": plan.method, "intent": plan.intent}]
        planner_note = getattr(self.planner, "last_llm_error", None)
        if plan.planner_method == "fallback" and planner_note:
            trace[0]["fallback_reason"] = planner_note

        if plan.method == "retrieval":
            applied_filters = filters if plan.scope in {"projects", "all"} else None
            after = len(self.retriever.candidate_indices(plan.scope, applied_filters))
            retrieval_evidence = self.retriever.search(
                plan.search_query, top_k=self.top_k, scope=plan.scope, filters=applied_filters,
            )
            # Mixed private-profile/project questions need both evidence sets;
            # keep the existing ranking unchanged within each scope.
            if (plan.scope == "all" and re.search(r"\bprivate\s+profile\b", question, re.I)
                    and re.search(r"\b(?:current\s+utas|public\s+(?:utas\s+)?projects?|project\s+corpus|advertised\s+projects?)\b", question, re.I)):
                project_rows = self.retriever.search(plan.search_query, top_k=self.top_k, scope="projects")
                seen = {(row.get("item_type"), row.get("project_id"), row.get("document_id")) for row in retrieval_evidence}
                retrieval_evidence.extend(row for row in project_rows
                                          if (row.get("item_type"), row.get("project_id"), row.get("document_id")) not in seen)
            trace.append({"step": "hybrid_retrieval", "scope": plan.scope,
                          "candidate_count_before": before, "candidate_count_after": after,
                          "result_count": len(retrieval_evidence)})
        elif plan.method == "graph":
            operation_used = plan.graph_operation
            graph_result = execute_graph_operation(self.graph, plan)
            after = _result_count(graph_result)
            trace.append({"step": "sparql_query", "operation": operation_used,
                          "candidate_count_before": before, "result_count": after,
                          "result_unit": "matching projects" if isinstance(graph_result, dict) and "projects" in graph_result
                          else "SPARQL result rows"})
        elif plan.method == "hybrid_graph":
            operation_used = plan.graph_operation
            graph_result = execute_graph_operation(self.graph, plan)
            if not isinstance(graph_result, list):
                raise ValueError("hybrid_graph operation must return project records")
            candidate_ids = {str(row["project_id"]) for row in graph_result if row.get("project_id")}
            after = len(candidate_ids)
            trace.append({"step": "sparql_candidate_selection", "operation": operation_used,
                          "candidate_count_before": before, "candidate_count_after": after,
                          "project_ids": sorted(candidate_ids)})
            retrieval_evidence = self.retriever.search(
                plan.search_query, top_k=self.top_k, scope="projects",
                candidate_project_ids=candidate_ids,
            )
            returned_ids = {str(row.get("project_id")) for row in retrieval_evidence if row.get("project_id")}
            if not returned_ids <= candidate_ids:
                raise RuntimeError("Hybrid retrieval returned a project outside the SPARQL candidate set")
            trace.append({"step": "hybrid_rank_graph_candidates", "candidate_project_count": len(candidate_ids),
                          "retrieval_result_count": len(retrieval_evidence),
                          "all_results_within_graph_candidates": returned_ids <= candidate_ids})
        else:  # ReasoningPlan validation makes this unreachable; retain a safe dispatch boundary.
            raise ValueError(f"Unsupported reasoning method: {plan.method!r}")

        sources, project_ids = _source_metadata(retrieval_evidence, graph_result)
        return {
            "original_question": question,
            "reasoning_method": plan.method,
            "scope": plan.scope,
            "planner_method": plan.planner_method,
            "interpreted_intent": plan.intent,
            "extracted_constraints": filters,
            "graph_operation_used": operation_used,
            "candidate_count_before": before,
            "candidate_count_after": after,
            "ranked_retrieval_evidence": retrieval_evidence,
            "graph_result": graph_result,
            "source_urls": sources,
            "project_ids": project_ids,
            "tool_trace": trace,
        }
