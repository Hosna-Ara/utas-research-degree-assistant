"""Generate answers strictly from evidence returned by ReasoningRouter."""

import json
import re
from typing import Protocol

import requests
from pydantic import ValidationError

from utas_research_assistant.config import ANSWER_MODEL, ANSWER_TIMEOUT_SECONDS, OLLAMA_HOST
from utas_research_assistant.generation.citations import build_citation_map, validate_citations
from utas_research_assistant.generation.evidence_guard import assess_evidence_capability
from utas_research_assistant.generation.models import AnswerResponse
from utas_research_assistant.generation.prompts import ANSWER_SYSTEM_PROMPT


class AnswerProvider(Protocol):
    model: str

    def generate(self, prompt: dict) -> str: ...


class OllamaAnswerProvider:
    def __init__(self, host: str = OLLAMA_HOST, model: str = ANSWER_MODEL,
                 timeout: float = ANSWER_TIMEOUT_SECONDS, session=None):
        self.host, self.model, self.timeout = host.rstrip("/"), model, timeout
        self.session = session or requests.Session()

    def generate(self, prompt: dict) -> str:
        response = self.session.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "system": ANSWER_SYSTEM_PROMPT,
                  "prompt": json.dumps(prompt, ensure_ascii=False),
                  "format": {"type": "object", "properties": {
                      "answer": {"type": "string"},
                      "insufficient_evidence": {"type": "boolean"}},
                      "required": ["answer", "insufficient_evidence"], "additionalProperties": False},
                  "stream": False, "think": False,
                  "options": {"temperature": 0.1, "num_predict": 512}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"]


def _trim_item(item: dict) -> dict:
    allowed = ("title", "text", "source_url", "project_id", "primary_supervisor",
               "research_categories", "degree_types", "student_types", "location",
               "funding_status", "status", "category", "provenance", "item_type",
               "supervisor_id", "canonical_name", "discovery_profile_id", "discovery_url", "document_id", "chunk_id",
               "school", "research_fields", "related_project_ids", "related_research_categories",
               "is_ict_supervisor", "orcid", "google_scholar_url", "local_filename", "source_path",
               "file_type")
    result = {key: item[key] for key in allowed if key in item}
    if isinstance(result.get("text"), str):
        result["text"] = result["text"][:2200]
    return result


def structured_evidence(evidence: dict, citation_map: dict[str, dict]) -> dict:
    retrieval = [_trim_item(row) | {"citation_id": key}
                 for key, row in citation_map.items() if row.get("item_type")]
    graph = evidence.get("graph_result")
    return {
        "reasoning_method": evidence.get("reasoning_method"),
        "graph_operation": evidence.get("graph_operation_used"),
        "retrieval_evidence": retrieval,
        "graph_evidence": graph,
        "graph_source_urls": [url for url in evidence.get("source_urls", []) if url],
        "project_ids_in_evidence": evidence.get("project_ids", []),
        "citation_ids": list(citation_map),
        "citation_catalog": [{"citation_id": identifier,
                              "title": item.get("title") or item.get("supervisor") or item.get("category"),
                              "source_url": item.get("source_url"),
                              "project_id": item.get("project_id"),
                              "item_type": item.get("item_type", item.get("evidence_type"))}
                             for identifier, item in citation_map.items()],
    }


def graph_tool_summary(evidence: dict) -> str | None:
    if evidence.get("reasoning_method") not in {"graph", "hybrid_graph"}:
        return None
    operation = evidence.get("graph_operation_used") or "SPARQL query"
    result = evidence.get("graph_result")
    if isinstance(result, dict) and "count" in result:
        return f"{result['count']} matching projects (operation: {operation})."
    if isinstance(result, list):
        unit = "supervisors" if operation.startswith("supervisors") else (
            "categories" if "category" in operation else "results")
        return f"{len(result)} {unit} returned (operation: {operation})."
    if result is None:
        return f"No matching graph result (operation: {operation})."
    if isinstance(result, dict) and result.get("project_id"):
        detail = result.get("title") or "Project record"
        supervisor = result.get("primary_supervisor")
        if supervisor:
            detail += f"; supervisor: {supervisor}"
        return f"{detail} (project {result['project_id']}; operation: {operation})."
    return f"A project record was returned (operation: {operation})."


def _clean_project_mentions(answer: str, project_ids: list[str]) -> str:
    allowed = set(map(str, project_ids))
    def replace(match):
        return match.group(0) if match.group(1) in allowed else "project"
    return re.sub(r"\bproject\s+#?(\d{4,})\b", replace, answer, flags=re.IGNORECASE)


def _topic_terms(text: str) -> set[str]:
    ignored = {"what", "which", "how", "many", "are", "the", "for", "with", "from", "that",
               "this", "and", "or", "do", "does", "i", "me", "my", "is", "in", "on", "a",
               "an", "to", "of", "available", "need", "show", "find", "looking", "about",
               "research", "degree", "degrees", "project", "projects", "student", "students",
               "when", "each", "am", "want", "can", "could"}
    terms = {term for term in re.findall(r"[a-z]+", text.casefold()) if term not in ignored}
    return {re.sub(r"ies$", "y", re.sub(r"ing$", "", re.sub(r"s$", "", term))) for term in terms}


def _is_relevant(question: str, answer: str, evidence: dict) -> bool:
    question_terms = _topic_terms(question)
    answer_terms = _topic_terms(answer)
    overlap = question_terms & answer_terms
    evidence_text = " ".join(
        str(row.get("text", "")) for row in evidence.get("ranked_retrieval_evidence", [])
    ).casefold()
    supported_terms = {term for term in question_terms if term in _topic_terms(evidence_text)}
    high_specificity = question_terms & {"probability", "gpa", "guaranteed"}
    if high_specificity and not (high_specificity & _topic_terms(evidence_text)):
        return False
    # Require two topical anchors for compound questions; one for a single-topic request.
    threshold = min(2, max(1, len(question_terms)))
    return len(overlap) >= threshold and len(supported_terms) >= threshold


def _deterministic_count_answer(evidence: dict, citation_map: dict[str, dict]) -> str | None:
    """Preserve exact SPARQL counts if a small model omits or alters them."""
    operation, result = evidence.get("graph_operation_used"), evidence.get("graph_result")
    citation = next(iter(citation_map), None)
    if operation == "count_open_funded_international_ict_projects" and isinstance(result, dict):
        return f"There are {int(result['count'])} matching open, funded international PhD projects in ICT" + (f" [{citation}]." if citation else ".")
    if operation == "open_projects_by_category" and isinstance(result, list):
        entries = "; ".join(f"{row['category']}: {int(row['count'])}" for row in result)
        return f"Open projects by research category: {entries}" + (f" [{citation}]." if citation else ".")
    return None


def _profile_excerpt(profile: dict) -> str | None:
    """Extract a short source-derived bio without exposing retrieval labels."""
    text = str(profile.get("bio") or profile.get("text") or "").strip()
    if not text:
        return None
    if "Bio:" in text:
        text = text.split("Bio:", 1)[1]
    text = re.split(r"\n\s*(?:Research fields|Positions|Degrees|Related advertised projects)\s*:", text, maxsplit=1, flags=re.I)[0]
    text = " ".join(text.split()).strip()
    if not text:
        return None
    return " ".join(re.split(r"(?<=[.!?])\s+", text)[:2])[:420].rstrip() + ("…" if len(text) > 420 else "")


def _deterministic_graph_answer(evidence: dict, citation_map: dict[str, dict]) -> str | None:
    """Render exhaustive SPARQL outputs directly so the model cannot truncate the result set."""
    operation, result = evidence.get("graph_operation_used"), evidence.get("graph_result")
    if evidence.get("scope") == "projects" and evidence.get("ranked_retrieval_evidence"):
        safe_answer = _safe_evidence_answer(evidence.get("original_question", ""), evidence, citation_map)
        if safe_answer:
            return safe_answer[0]
    graph_citation = next((identifier for identifier, item in citation_map.items()
                           if item.get("evidence_type") == "graph_result"), None)
    if operation == "project_by_id" and isinstance(result, dict):
        identifier = str(result.get("project_id", ""))
        citation = next((key for key, item in citation_map.items()
                         if str(item.get("project_id", "")) == identifier and item.get("source_url")), graph_citation)
        supervisor = result.get("primary_supervisor")
        title = result.get("title", "")
        answer = f"Project {identifier} is {title!r}."
        if supervisor:
            answer += f" It lists {supervisor} as supervisor."
        profile = result.get("supervisor_profile")
        if isinstance(profile, dict):
            fields = profile.get("research_fields") or []
            background = bool(re.search(r"\b(?:research background|background|research profile|interests?)\b", evidence.get("original_question", ""), re.I))
            if fields:
                profile_citation = next((key for key, item in citation_map.items()
                                         if item.get("source_url") == profile.get("source_url")
                                         or item.get("source_url") == profile.get("discovery_url")), graph_citation)
                answer += f" Their listed research fields include {', '.join(fields)}."
                if profile_citation:
                    answer += f" [{profile_citation}]"
            if background:
                if profile.get("title"):
                    answer += f" Their role is listed as {profile['title']}"
                if profile.get("school"):
                    answer += f" at {profile['school']}"
                excerpt = _profile_excerpt(profile)
                if excerpt:
                    answer += f" The profile describes: {excerpt}"
        return answer + (f" [{citation}]" if citation else "")
    if operation == "open_projects_by_category" and isinstance(result, list):
        entries = "; ".join(f"{row['category']}: {int(row['count'])}" for row in result)
        return f"Open projects by research category: {entries}" + (f" [{graph_citation}]" if graph_citation else "")
    if operation == "count_open_funded_international_ict_projects" and isinstance(result, dict):
        return f"There are {int(result['count'])} matching open, funded international PhD projects in ICT" + (f" [{graph_citation}]" if graph_citation else "")
    if operation in {"supervisor_research_fields", "supervisor_school", "supervisor_by_name"} and isinstance(result, dict):
        name = result.get("canonical_name") or result.get("supervisor") or "This supervisor"
        details = []
        fields = result.get("research_fields") or []
        if fields:
            details.append("listed research fields include " + ", ".join(map(str, fields)))
        if result.get("school"):
            details.append(f"is affiliated with {result['school']}")
        if result.get("title"):
            details.insert(0, str(result["title"]))
        if not details and result.get("bio"):
            details.append(str(result["bio"])[:700].rstrip() + ("…" if len(str(result["bio"])) > 700 else ""))
        if details:
            sentence = f"{name} is listed as " + "; ".join(details)
            excerpt = _profile_excerpt(result)
            if excerpt and re.search(r"\b(?:tell me|about|background|profile)\b", evidence.get("original_question", ""), re.I):
                sentence += f" The profile describes: {excerpt}"
            return sentence + (f" [{graph_citation}]" if graph_citation else "")
        return f"A profile for {name} was found, but it does not contain the requested detail." + (f" [{graph_citation}]" if graph_citation else "")
    if operation in {"supervisors_by_research_field", "ict_supervisors", "ict_supervisors_by_research_field", "supervisors_with_orcid"} and isinstance(result, list):
        label = ("ICT supervisors matching the requested research field" if operation == "ict_supervisors_by_research_field"
                 else "supervisors matching the requested research field" if operation == "supervisors_by_research_field"
                 else "ICT supervisors" if operation == "ict_supervisors" else "supervisors with ORCID profiles")
        names = "; ".join(
            f"{row.get('supervisor')}" + (f" ({row['school']})" if row.get("school") else "")
            + (f" — {row['orcid']}" if row.get("orcid") else "")
            for row in result
        )
        return f"{len(result)} {label}: {names}" + (f" [{graph_citation}]" if graph_citation else "")
    if operation == "categories_with_both_degree_types" and isinstance(result, list):
        return ("Categories containing both PhD and Master by Research projects: " + ", ".join(result) +
                (f" [{graph_citation}]" if graph_citation else ""))
    if operation in {"supervisors_with_multiple_projects", "supervisors_with_funded_ict_international_projects",
                     "supervisors_across_multiple_categories"} and isinstance(result, list):
        label = {"supervisors_with_multiple_projects": "supervisors with more than one advertised project",
                 "supervisors_with_funded_ict_international_projects": "supervisors with funded ICT projects accepting international students",
                 "supervisors_across_multiple_categories": "supervisors with projects across multiple research categories"}[operation]
        field = "project_count" if operation != "supervisors_across_multiple_categories" else "category_count"
        details = "; ".join(f"{row['supervisor']} ({row[field]})" for row in result)
        return (f"{len(result)} {label}: {details}" + (f" [{graph_citation}]" if graph_citation else ""))
    if operation == "multi_constraint_projects" and evidence.get("reasoning_method") == "hybrid_graph":
        safe_answer = _safe_evidence_answer(evidence.get("original_question", ""), evidence, citation_map)
        return safe_answer[0] if safe_answer else None
    if operation in {"multi_constraint_projects", "projects_by_supervisor"} and isinstance(result, list):
        matches = []
        for row in result:
            identifier = str(row.get("project_id", ""))
            citation = next((key for key, item in citation_map.items()
                             if str(item.get("project_id", "")) == identifier and item.get("source_url")), graph_citation)
            matches.append(f"{row.get('title', 'Project')} (project {identifier})" + (f" [{citation}]" if citation else ""))
        return f"{len(matches)} potential project matches: " + "; ".join(matches)
    return None


def _evidence_citation(row: dict, citation_map: dict[str, dict]) -> str | None:
    for identifier, item in citation_map.items():
        if item.get("item_type") == row.get("item_type") and item.get("title") == row.get("title") and item.get("text") == row.get("text"):
            return identifier
    project_id = row.get("project_id")
    if project_id:
        return next((identifier for identifier, item in citation_map.items()
                     if str(item.get("project_id", "")) == str(project_id) and item.get("source_url")), None)
    return None


def _safe_evidence_answer(question: str, evidence: dict, citation_map: dict[str, dict]) -> tuple[str, list[str]] | None:
    """Return a short extractive/general or project shortlist when model prose is unusable."""
    rows = evidence.get("ranked_retrieval_evidence", []) or []
    project_rows = rows and all(row.get("item_type") == "research_project" for row in rows)
    local_rows = [row for row in rows if row.get("item_type") == "local_document"]
    if local_rows:
        query_terms = _topic_terms(question)
        def local_relevance(row: dict) -> tuple[int, int]:
            searchable = " ".join(str(row.get(key) or "") for key in ("document_id", "title", "local_filename", "text"))
            overlap = len(query_terms & _topic_terms(searchable))
            # Prefer lexical evidence from the question while retaining the
            # stable hybrid order for ties; no score boosting is applied.
            return overlap, -int(row.get("rank") or 0)
        row = max(local_rows, key=local_relevance)
        citation = _evidence_citation(row, citation_map)
        if citation:
            text = " ".join(str(row.get("text", "")).split())
            question_terms = _topic_terms(question)
            sentences = re.split(r"(?<=[.!?])\s+", text)
            relevant = [sentence for sentence in sentences if question_terms & _topic_terms(sentence)]
            if re.search(r"\b(?:matching|factor|priority|priorities)\b", question, re.I):
                excerpt = " ".join((relevant or sentences)[:5])
            elif re.search(r"\b(?:why|reason|rejected|declined)\b", question, re.I):
                # Include the source sentence and its nearby context without
                # naming any private journal entry or persona label.
                sentence_terms = [_topic_terms(sentence) for sentence in sentences]
                term_frequency = {term: sum(term in current for current in sentence_terms)
                                  for term in question_terms}
                distinctive = {term for term, count in term_frequency.items() if count == 1}
                anchor = max(range(len(sentences)), key=lambda index: (
                    len(distinctive & sentence_terms[index]),
                    len(question_terms & sentence_terms[index]),
                    -index,
                )) if sentences else 0
                excerpt = " ".join(sentences[anchor:anchor + 3])
            else:
                excerpt = " ".join((relevant or sentences)[:3]).strip()[:700]
            local_label = row.get("document_id") or row.get("title") or "private local document"
            answer = f"According to the private local document {local_label}: {excerpt} [{citation}]"
            if (re.search(r"\b(?:match|fit|suitab|recommend)", question, re.I)
                    and re.search(r"\b(?:controlled mismatch|poor fit|no strong match|weak match)", text, re.I)):
                answer = "The local evidence does not establish a strong match under the stated constraints. " + answer
            if re.search(r"\b(?:current\s+utas|projects?|opportunities)\b", question, re.I):
                project_rows_mixed = [item for item in rows if item.get("item_type") == "research_project"]
                matches = []
                used = [citation]
                for item in project_rows_mixed[:5]:
                    project_citation = _evidence_citation(item, citation_map)
                    if project_citation:
                        matches.append(f"{item.get('title', 'Project')} (project {item.get('project_id')}) [{project_citation}]")
                        used.append(project_citation)
                if matches:
                    answer += " Potential public-corpus matches to review include: " + "; ".join(matches) + "."
                    return answer, list(dict.fromkeys(used))
            return answer, [citation]
    supervisor_rows = [row for row in rows if row.get("item_type") == "supervisor_profile"]
    if supervisor_rows:
        row = supervisor_rows[0]
        citation = _evidence_citation(row, citation_map)
        name = row.get("canonical_name") or row.get("title") or "The supervisor"
        if citation:
            fields = row.get("research_fields") or []
            school = row.get("school")
            if fields and ("interest" in question.casefold() or "field" in question.casefold() or "ai" in question.casefold()):
                return f"{name}'s profile lists these research fields: {', '.join(map(str, fields))}. [{citation}]", [citation]
            excerpt = _profile_excerpt(row)
            if excerpt:
                role = ""
                role_match = re.search(r"(?:^|\n)Role:\s*([^\n]+)", str(row.get("text", "")), re.I)
                if role_match:
                    role = role_match.group(1).strip()
                elif row.get("title") and str(row.get("title")) != str(name):
                    role = row.get("title") or ""
                school = row.get("school") or ""
                details = ", ".join(part for part in (role, school) if part)
                lead = f"{name}"
                if details:
                    lead += f" is listed as {details}."
                return f"{lead} The profile describes: {excerpt} [{citation}]", [citation]
            if school:
                return f"{name} is affiliated with {school}. [{citation}]", [citation]
    if project_rows and (evidence.get("scope") == "projects" or evidence.get("reasoning_method") == "hybrid_graph"):
        parts, citations = [], []
        for row in rows[:5]:
            citation = _evidence_citation(row, citation_map)
            project_id = row.get("project_id")
            if not citation or not project_id:
                continue
            parts.append(f"{row.get('title', 'Project')} (project {project_id}) [{citation}]")
            citations.append(citation)
        if parts:
            intro = "Potential matches among the graph-filtered projects include: " if evidence.get("reasoning_method") == "hybrid_graph" else "Potential matches from ranked project results include: "
            return intro + "; ".join(parts) + ".", citations

    terms = _topic_terms(question)
    evidence_text = " ".join(str(row.get("text", "")) for row in rows).casefold()
    high_specificity = terms & {"probability", "gpa", "guaranteed"}
    if high_specificity and not (high_specificity & _topic_terms(evidence_text)):
        return None
    candidates = []
    for index, row in enumerate(rows[:5]):
        citation = _evidence_citation(row, citation_map)
        if not citation:
            continue
        text = str(row.get("text", ""))
        if "english" in terms and "score" in terms and "minimum overall score" in text.casefold():
            lower = text.casefold()
            start = lower.rfind("ielts", 0, lower.find("minimum overall score"))
            if start < 0:
                start = lower.find("minimum overall score")
            end = lower.find("\nOR", lower.find("minimum overall score", start))
            excerpt = text[start:end if end > start else min(len(text), start + 1100)].strip()
            lines = [" ".join(line.split()) for line in excerpt.splitlines() if line.strip() and line.strip().lower() not in {"or", "minimum additional scores"}]
            return "English-language scores listed by UTAS:\n" + "\n".join(f"- {line}" for line in lines) + f"\n[{citation}]", [citation]
        if ("document" in terms or "apply" in terms) and "at a minimum you will need" in text.casefold():
            lower = text.casefold()
            start = lower.find("at a minimum you will need")
            end = lower.find("once you have prepared your documents", start)
            excerpt = text[start:end if end > start else min(len(text), start + 1200)].strip()
            lines = [" ".join(line.lstrip("-• ").split()) for line in excerpt.splitlines() if line.strip()]
            return "The application documents listed in the local UTAS guidance include:\n" + "\n".join(f"- {line}" for line in lines) + f"\n[{citation}]", [citation]
    if "scholarship" in terms or "stipend" in terms:
        excerpts, used, seen = [], [], set()
        for row in rows[:5]:
            citation = _evidence_citation(row, citation_map)
            if not citation:
                continue
            text = str(row.get("text", ""))
            pattern = r"China Scholarship Council|CSC Scholarship|\$\s?[\d,]+(?:\.\d+)?\s?(?:pa|per annum)?"
            for match in re.finditer(pattern, text, re.I):
                start, end = max(0, match.start() - 170), min(len(text), match.end() + 210)
                prefix = text[start:match.start()]
                boundary = max(prefix.rfind("\n"), prefix.rfind(". "))
                if boundary >= 0:
                    start += boundary + (2 if prefix[boundary:boundary + 2] == ". " else 1)
                snippet = " ".join(text[start:end].split())
                if snippet not in seen:
                    seen.add(snippet)
                    excerpts.append(f"“{snippet}” [{citation}]")
                    used.append(citation)
                if len(excerpts) >= 3:
                    break
            if len(excerpts) >= 3:
                break
        if excerpts:
            return "The local scholarship material includes these details: " + "; ".join(excerpts), list(dict.fromkeys(used))
    if "apply" in terms or "application" in terms:
        for row in rows[:5]:
            citation = _evidence_citation(row, citation_map)
            if not citation:
                continue
            text = str(row.get("text", ""))
            guidance = next((segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", text)
                             if "allow plenty of time" in segment.casefold()
                             and "submit an application" in segment.casefold()
                             and "closing date" in segment.casefold()), None)
            if guidance:
                return f"The research-degree application guidance says: “{guidance}” [{citation}]", [citation]
        for phrase in ("submit your application", "online application system", "how to apply"):
            for row in rows[:5]:
                citation = _evidence_citation(row, citation_map)
                if not citation:
                    continue
                text = str(row.get("text", ""))
                lower = text.casefold()
                position = lower.find(phrase)
                if position < 0:
                    continue
                start = max(0, text.rfind("\n", 0, position))
                end = text.find("\n\n", position)
                excerpt = " ".join(text[start:end if end >= 0 else min(len(text), position + 550)].split())
                if excerpt:
                    return f"The research-degree application information says: “{excerpt}” [{citation}]", [citation]
    for index, row in enumerate(rows[:5]):
        citation = _evidence_citation(row, citation_map)
        if not citation:
            continue
        text = str(row.get("text", ""))
        segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", text) if segment.strip()]
        for segment in segments:
            overlap = terms & _topic_terms(segment)
            if overlap:
                candidates.append((len(overlap), -index, len(segment), segment, citation, row.get("title", "UTAS source")))
    if not candidates:
        return None
    _, _, _, segment, citation, title = max(candidates)
    excerpt = segment[:650].rstrip()
    if len(segment) > 650:
        excerpt += "…"
    return f"The local UTAS material states: “{excerpt}” [{citation}]", [citation]


class AnswerGenerator:
    def __init__(self, provider: AnswerProvider | None = None):
        self.provider = provider or OllamaAnswerProvider()
        self.model = getattr(self.provider, "model", ANSWER_MODEL)
        self.last_error: str | None = None
        self.last_generation_method: str | None = None
        self.last_model_called = False

    def generate(self, question: str, evidence: dict) -> AnswerResponse:
        """Answer using supplied router output only; generation failure stays explicit."""
        citation_map, sources = build_citation_map(evidence)
        self.last_error = None
        self.last_model_called = False
        ambiguity = evidence.get("supervisor_ambiguity") or []
        if ambiguity:
            names = ", ".join(str(name) for name in ambiguity)
            self.last_generation_method = "deterministic_disambiguation"
            return AnswerResponse(
                question=question,
                answer=f"I found more than one supervisor matching that first name: {names}. Please provide the supervisor's full name so I can identify the correct projects.",
                reasoning_method=evidence.get("reasoning_method", "graph"),
                planner_method=evidence.get("planner_method", "fallback"),
                citations=[], sources=[], tool_used=None, tool_result=None,
                insufficient_evidence=False, project_ids=[], generation_model=self.model,
            )
        gap = assess_evidence_capability(question, evidence)
        if gap:
            citation = next((identifier for identifier, item in citation_map.items()
                             if gap.project_id and str(item.get("project_id", "")) == gap.project_id
                             and item.get("source_url")), None)
            answer = gap.message + (f" [{citation}]" if citation else "")
            self.last_generation_method = "insufficient_evidence_fallback"
            return AnswerResponse(
                question=question, answer=answer,
                reasoning_method=evidence.get("reasoning_method", "retrieval"),
                planner_method=evidence.get("planner_method", "fallback"),
                citations=[citation] if citation else [],
                sources=[source for source in sources if source["citation_id"] == citation] if citation else [],
                tool_used="SPARQL Knowledge Graph" if evidence.get("reasoning_method") in {"graph", "hybrid_graph"} else None,
                tool_result=graph_tool_summary(evidence), insufficient_evidence=True,
                project_ids=list(dict.fromkeys(map(str, evidence.get("project_ids", [])))),
                generation_model=self.model,
            )
        prompt = {"question": question,
                  "evidence": structured_evidence(evidence, citation_map)}
        generation_method = "ollama"
        try:
            self.last_model_called = True
            raw = self.provider.generate(prompt)
            payload = json.loads(raw)
            if not isinstance(payload, dict) or set(payload) != {"answer", "insufficient_evidence"}:
                raise ValueError("Model response must contain only answer and insufficient_evidence")
            if not isinstance(payload["answer"], str) or not isinstance(payload["insufficient_evidence"], bool):
                raise ValueError("Malformed answer response types")
            model_answer = re.sub(r"\((S\d+)\)", r"[\1]", payload["answer"])
            answer, citations = validate_citations(model_answer, citation_map)
            insufficient = payload["insufficient_evidence"]
            exact_graph_answer = _deterministic_graph_answer(evidence, citation_map)
            if exact_graph_answer:
                answer = exact_graph_answer
                citations = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", answer)))
                insufficient = False
                generation_method = "deterministic_evidence_render"
            elif insufficient:
                safe_answer = _safe_evidence_answer(question, evidence, citation_map)
                if safe_answer:
                    answer, citations = safe_answer
                    insufficient = False
                    generation_method = "extractive_or_ranked_evidence_fallback"
                else:
                    answer = "The available local knowledge does not contain enough information to answer this reliably."
                    citations = []
                    generation_method = "insufficient_evidence_fallback"
            elif evidence.get("ranked_retrieval_evidence") and not _is_relevant(question, answer, evidence):
                safe_answer = _safe_evidence_answer(question, evidence, citation_map)
                if safe_answer:
                    answer, citations = safe_answer
                    generation_method = "extractive_or_ranked_evidence_fallback"
                else:
                    answer = "The available local knowledge does not contain enough information to answer this reliably."
                    citations, insufficient = [], True
                    generation_method = "insufficient_evidence_fallback"
            else:
                exact_count_answer = _deterministic_count_answer(evidence, citation_map)
                if exact_count_answer:
                    answer = exact_count_answer
                    cited = re.search(r"\[(S\d+)\]", answer)
                    citations = [cited.group(1)] if cited else []
                elif not citations and not insufficient:
                    safe_answer = _safe_evidence_answer(question, evidence, citation_map)
                    if safe_answer:
                        answer, citations = safe_answer
                        generation_method = "extractive_or_ranked_evidence_fallback"
                    else:
                        answer = "The available local knowledge does not contain enough information to answer this reliably."
                        citations, insufficient = [], True
                        generation_method = "insufficient_evidence_fallback"
            project_ids = list(dict.fromkeys(map(str, evidence.get("project_ids", []))))
            answer = _clean_project_mentions(answer, project_ids)
            if not answer:
                raise ValueError("Empty answer")
        except (requests.RequestException, ValueError, KeyError, TypeError, ValidationError) as exc:
            self.last_error = f"Answer generation unavailable or invalid: {exc}"
            exact_graph_answer = _deterministic_graph_answer(evidence, citation_map)
            safe_answer = _safe_evidence_answer(question, evidence, citation_map)
            if exact_graph_answer:
                answer = exact_graph_answer
                citations = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", answer)))
                insufficient = False
                generation_method = "deterministic_evidence_render"
            elif safe_answer:
                answer, citations = safe_answer
                insufficient = False
                generation_method = "extractive_or_ranked_evidence_fallback"
            else:
                answer = "The available local knowledge does not contain enough information to answer this reliably."
                citations, insufficient = [], True
                generation_method = "insufficient_evidence_fallback"
        else:
            self.last_error = None
        self.last_generation_method = generation_method

        method = evidence.get("reasoning_method", "retrieval")
        return AnswerResponse(
            question=question, answer=answer, reasoning_method=method,
            planner_method=evidence.get("planner_method", "fallback"), citations=citations,
            sources=[source for source in sources if source["citation_id"] in citations],
            tool_used="SPARQL Knowledge Graph" if method in {"graph", "hybrid_graph"} else None,
            tool_result=graph_tool_summary(evidence), insufficient_evidence=insufficient,
            project_ids=list(dict.fromkeys(map(str, evidence.get("project_ids", [])))),
            generation_model=self.model,
        )
