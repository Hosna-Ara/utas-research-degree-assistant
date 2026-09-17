"""Deterministic evidence references and citation validation."""

import re


def _evidence_items(evidence: dict) -> list[dict]:
    items = []
    for row in evidence.get("ranked_retrieval_evidence", []) or []:
        item = dict(row)
        item["evidence_type"] = item.get("item_type", "retrieval")
        items.append(item)

    graph_result = evidence.get("graph_result")
    if graph_result is not None:
        items.append({"title": f"SPARQL result: {evidence.get('graph_operation_used') or 'project lookup'}",
                      "text": graph_result, "source_url": None, "evidence_type": "graph_result"})

    def add_graph(value):
        if isinstance(value, dict):
            # A graph result may be a single project, a wrapper with projects, or an aggregate.
            if value.get("source_url") or value.get("project_id") or value.get("title"):
                items.append({**value, "evidence_type": "graph"})
            for key, child in value.items():
                if key not in {"source_url", "project_id", "title"}:
                    add_graph(child)
        elif isinstance(value, list):
            for child in value:
                add_graph(child)

    add_graph(evidence.get("graph_result"))
    return items


def build_citation_map(evidence: dict) -> tuple[dict[str, dict], list[dict]]:
    """Return stable S-n references for actual evidence items with usable provenance."""
    citation_map, sources = {}, []
    seen = set()
    for item in _evidence_items(evidence):
        url = item.get("source_url")
        # Graph/tool evidence can be cited by operation even without an individual page URL.
        identity = (url, item.get("project_id"), item.get("title"),
                    repr(item.get("text")))
        if identity in seen:
            continue
        seen.add(identity)
        citation_id = f"S{len(citation_map) + 1}"
        source = {
            "citation_id": citation_id,
            "title": (item.get("canonical_name") if (item.get("item_type") == "supervisor_profile"
                                                       or item.get("canonical_name"))
                      else item.get("title")) or _source_title(item),
            "source_url": url,
            "project_id": item.get("project_id"),
            "item_type": item.get("item_type", item.get("evidence_type")),
            "document_id": item.get("document_id"),
            "provenance": item.get("provenance", []),
            "supervisor_profile": item.get("supervisor_profile"),
            "local_filename": item.get("local_filename"),
            "source_path": item.get("source_path"),
            "file_type": item.get("file_type"),
        }
        citation_map[citation_id] = item
        sources.append(source)
    return citation_map, sources


def _source_title(item: dict) -> str:
    if item.get("project_id"):
        return f"Project {item['project_id']}"
    return item.get("supervisor") or item.get("category") or item.get("title") or item.get("document_id") or item.get("evidence_type", "Evidence")


def validate_citations(answer: str, citation_map: dict[str, dict]) -> tuple[str, list[str]]:
    """Remove unknown [S#] tokens and return the valid IDs used in the answer."""
    valid = set(citation_map)
    found = re.findall(r"\[(S\d+)\]", answer)
    answer = re.sub(r"\[(S\d+)\]", lambda match: match.group(0) if match.group(1) in valid else "", answer)
    used = list(dict.fromkeys(identifier for identifier in found if identifier in valid))
    return re.sub(r"[ \t]{2,}", " ", answer).strip(), used
