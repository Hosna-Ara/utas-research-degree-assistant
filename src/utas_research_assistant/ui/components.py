"""Small presentation components for the Streamlit shell.

The components consume the existing AnswerResponse/evidence contract. They do
not perform retrieval, planning, or answer generation.
"""

from html import escape

import streamlit as st

from utas_research_assistant.generation.models import AnswerResponse
from utas_research_assistant.ui.formatters import (
    metadata_value,
    official_source_url,
    official_supervisor_url,
    project_cards_from_sources,
    source_display_title,
)


LIST_GRAPH_OPERATIONS = {
    "supervisors_by_research_field", "ict_supervisors_by_research_field", "ict_supervisors",
    "supervisors_with_orcid", "supervisors_with_multiple_projects",
    "supervisors_with_funded_ict_international_projects", "supervisors_across_multiple_categories",
}


def _source_type(source: dict) -> str:
    value = source.get("item_type") or source.get("evidence_type") or "source"
    return {"research_project": "Research project", "supervisor_profile": "Supervisor profile",
            "general_chunk": "UTAS general information", "local_document": "Local document",
            "graph_result": "Knowledge Graph result"}.get(value, str(value).replace("_", " ").title())


def render_sources(response: AnswerResponse, project_ids_with_cards: set[str] | None = None) -> None:
    if not response.sources:
        return
    project_ids_with_cards = project_ids_with_cards or set()
    with st.expander(f"Evidence & sources ({len(response.sources)})", expanded=False):
        for source in response.sources:
            citation = escape(str(source.get("citation_id", "S?")))
            title = escape(source_display_title(source))
            source_type = escape(_source_type(source))
            document_id = escape(str(source.get("document_id") or ""))
            identifier = f'<span class="source-type">{document_id}</span>' if document_id else ""
            st.markdown(f'<div class="source-entry"><strong>[{citation}] {title}</strong><span class="source-type">{source_type}</span>{identifier}</div>', unsafe_allow_html=True)
            url = official_source_url(source.get("source_url")) or official_supervisor_url(source.get("source_url"))
            if url:
                if str(source.get("project_id", "")) in project_ids_with_cards:
                    st.caption("Official source link is shown on the project card.")
                else:
                    st.markdown(f'<a class="source-link" href="{escape(url)}" target="_blank" rel="noreferrer">Official source ↗</a>', unsafe_allow_html=True)
            if source.get("text"):
                st.caption("Evidence text available in the grounded answer context.")


def _pill(label: str, value: str) -> str:
    return f'<span class="project-pill"><b>{escape(label)}</b> {escape(value)}</span>'


def render_project_cards(response: AnswerResponse, graph_result: object = None) -> set[str]:
    projects = project_cards_from_sources(response.sources)
    # Graph project rows are already structured evidence. This presentation-only
    # adapter keeps them visible when a deterministic generation fallback returns
    # graph rows without citation source objects.
    if not projects and isinstance(graph_result, list):
        projects = [row for row in graph_result if isinstance(row, dict) and row.get("project_id")]
    if not projects:
        return set()
    st.markdown("<div class=\"result-heading\">Research projects</div>", unsafe_allow_html=True)
    for index, project in enumerate(projects, start=1):
        title = escape(str(project.get("title") or "Research project"))
        project_id = escape(str(project.get("project_id") or ""))
        pills = []
        for label, key in (("", "degree_types"), ("", "student_types"), ("", "location"), ("", "funding_status"), ("", "status")):
            value = metadata_value(project.get(key))
            if value:
                pills.append(_pill(label, value))
        area = metadata_value(project.get("research_categories"))
        supervisor = metadata_value(project.get("primary_supervisor"))
        closing = metadata_value(project.get("closing_date"))
        details = []
        if area:
            details.append(f"<div><span>Research area</span><strong>{escape(area)}</strong></div>")
        if supervisor:
            details.append(f"<div><span>Supervisor</span><strong>{escape(supervisor)}</strong></div>")
        if closing:
            details.append(f"<div><span>Closing date</span><strong>{escape(closing)}</strong></div>")
        url = official_source_url(project.get("source_url"))
        profile = project.get("supervisor_profile") or {}
        profile_url = official_supervisor_url(profile.get("discovery_url") or profile.get("source_url"))
        profile_name = profile.get("canonical_name") or supervisor
        profile_fields = profile.get("research_fields") or []
        profile_html = ""
        if profile_name or profile.get("school") or profile_fields:
            fields = ", ".join(map(str, profile_fields[:4]))
            profile_html = '<div class="supervisor-mini"><div class="supervisor-heading">Supervisor profile</div>'
            if profile_name:
                profile_html += f'<strong>{escape(str(profile_name))}</strong>'
            if profile.get("title") and profile.get("title") not in str(profile_name):
                profile_html += f'<span>{escape(str(profile["title"]))}</span>'
            if profile.get("school"):
                profile_html += f'<span>{escape(str(profile["school"]))}</span>'
            if fields:
                profile_html += f'<span class="supervisor-fields">{escape(fields)}</span>'
            profile_html += '</div>'
        links = []
        if url:
            links.append(f'<a class="project-link" href="{escape(url)}" target="_blank" rel="noreferrer">View official source ↗</a>')
        if profile_url:
            links.append(f'<a class="project-link" href="{escape(profile_url)}" target="_blank" rel="noreferrer">View supervisor profile ↗</a>')
        html = (
            '<div class="project-result">'
            f'<div class="project-top"><span class="project-rank">#{index}</span><div><div class="project-title">{title}</div><div class="project-id">Project {project_id}</div></div></div>'
            f'<div class="project-pills">{"".join(pills)}</div>'
            f'<div class="project-details">{"".join(details)}</div>'
            f'<div class="project-card-lower"><div class="project-footer">{"<br>".join(links)}</div>{profile_html}</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
    return {str(project.get("project_id")) for project in projects if project.get("project_id")}


def render_graph_panel(response: AnswerResponse, evidence: dict) -> None:
    if not response.tool_used:
        return
    st.markdown('<div class="graph-badge">◈ Knowledge Graph used</div>', unsafe_allow_html=True)
    with st.expander("Knowledge Graph reasoning", expanded=False):
        st.markdown(f"**Tool**  {response.tool_used}")
        if response.tool_result:
            st.markdown(f"**Result**  {response.tool_result}")
        if response.reasoning_method == "hybrid_graph":
            before = evidence.get("candidate_count_before")
            after = evidence.get("candidate_count_after")
            if before is not None and after is not None:
                st.caption(f"Graph filtering: {before} → {after} eligible projects")
            st.caption("Hybrid ranking applied after graph filtering.")


def render_evidence_expander(evidence: dict) -> None:
    rows = evidence.get("ranked_retrieval_evidence", []) or []
    if not rows:
        return
    with st.expander("View retrieved evidence", expanded=False):
        for row in rows:
            st.markdown(f"**{row.get('title') or 'Evidence'}**")
            st.write(row.get("text", ""))
            url = official_source_url(row.get("source_url"))
            if url:
                st.markdown(f'<a class="source-link" href="{escape(url)}" target="_blank" rel="noreferrer">Official UTAS source ↗</a>', unsafe_allow_html=True)


def render_structured_graph_result(operation: str | None, result: object) -> None:
    """Present inherently list-shaped graph results without a prose dump."""
    if operation not in LIST_GRAPH_OPERATIONS or not isinstance(result, list):
        return
    rows = [row for row in result if isinstance(row, dict)]
    if not rows:
        return
    label = "supervisors found"
    st.markdown(f'<div class="structured-heading">{len(rows)} {label}</div>', unsafe_allow_html=True)
    def row_html(row: dict, index: int) -> str:
        name = escape(str(row.get("supervisor") or row.get("canonical_name") or "Supervisor"))
        detail = []
        if row.get("school"):
            detail.append(escape(str(row["school"])))
        if row.get("orcid"):
            detail.append("ORCID: " + escape(str(row["orcid"])))
        if row.get("project_count"):
            detail.append(f"{row['project_count']} advertised projects")
        if row.get("category_count"):
            detail.append(f"{row['category_count']} research categories")
        return f'<div class="structured-row"><b>{index}. {name}</b>' + (f'<span>{" · ".join(detail)}</span>' if detail else '') + '</div>'
    visible = rows[:10]
    st.markdown("".join(row_html(row, i) for i, row in enumerate(visible, 1)), unsafe_allow_html=True)
    if len(rows) > 10:
        with st.expander(f"Show all {len(rows)} supervisors", expanded=False):
            st.markdown("".join(row_html(row, i) for i, row in enumerate(rows[10:], 11)), unsafe_allow_html=True)


def render_assistant_result(payload: dict) -> None:
    if payload.get("error"):
        st.info(payload["error"])
        return
    response = AnswerResponse.model_validate(payload["response"])
    evidence = payload.get("evidence") or {}
    rows = evidence.get("ranked_retrieval_evidence", []) or []
    graph_result = evidence.get("graph_result")
    if response.insufficient_evidence:
        st.markdown('<div class="evidence-warning"><strong>Information not available in the current knowledge snapshot</strong><br>Some evidence needed for this question is not present in the local dataset.</div>', unsafe_allow_html=True)
    elif not rows and graph_result is None:
        st.markdown('<div class="no-match"><strong>No matching projects or local knowledge sources found</strong><br>Try broadening your research keywords or removing one eligibility constraint.</div>', unsafe_allow_html=True)
    if payload.get("generation_notice"):
        st.caption(payload["generation_notice"])
    operation = evidence.get("graph_operation_used")
    project_ids = {str(source.get("project_id")) for source in response.sources if source.get("project_id")}
    if operation in LIST_GRAPH_OPERATIONS and isinstance(graph_result, list):
        st.markdown(f"**{len(graph_result)} supervisors found.**")
        render_structured_graph_result(operation, graph_result)
    else:
        st.markdown(response.answer)
    project_ids |= render_project_cards(response, graph_result)
    render_sources(response, project_ids)
    render_graph_panel(response, evidence)
    st.markdown(f'<span class="method-badge">{escape(response.reasoning_method.replace("_", " ").title())}</span>', unsafe_allow_html=True)
