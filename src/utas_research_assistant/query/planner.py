"""Local Ollama query planner with a deterministic explicit-constraint fallback."""

import json
import re
from typing import Protocol

import requests
from pydantic import ValidationError

from utas_research_assistant.config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS
from utas_research_assistant.retrieval.filters import canonical
from utas_research_assistant.retrieval.local_documents import has_local_document_reference
from utas_research_assistant.query.models import ReasoningPlan, RetrievalPlan

SYSTEM_PROMPT = """You plan retrieval for an unofficial UTAS research-degree assistant.
Classify the question as general HDR information, research-project discovery, or both.
Extract only explicit structured constraints from the question. Never infer or invent
degree, student type, location, funding, status, category, supervisor, or project ID.
Keep the user's research-interest words in search_query. Return only valid JSON that
matches the supplied RetrievalPlan JSON schema, including planner_method='llm'.
Use scope 'general' for policy/application/entry/scholarship information, 'projects'
for finding or asking about specific projects, and 'all' only when both are needed.
Project-discovery wording includes asking whether a research project exists "about X",
"on X", or "related to X", asking for projects/opportunities in an interest area, or
asking what projects cover a topic. Route those to projects and retain X in search_query.
Examples: "Is there a research project about phishing detection?" uses
scope=projects, search_query="phishing detection"; "What English score do I need for a
PhD?" uses scope=general."""

CONSTRAINT_FIELDS = (
    "degree_type", "student_type", "location", "funding_status", "status",
    "research_category", "supervisor", "project_id",
)


class PlannerProvider(Protocol):
    def is_available(self) -> bool: ...
    def generate(self, question: str) -> str: ...


class OllamaPlanner:
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL,
                 timeout: float = OLLAMA_TIMEOUT_SECONDS, session=None):
        self.host, self.model, self.timeout = host.rstrip("/"), model, timeout
        self.session = session or requests.Session()
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=min(self.timeout, 2))
            response.raise_for_status()
            names = {entry.get("name") for entry in response.json().get("models", [])}
            self._available = self.model in names or any(
                name and name.split(":")[0] == self.model.split(":")[0] for name in names
            )
        except (requests.RequestException, ValueError, AttributeError):
            self._available = False
        return self._available

    def generate(self, question: str) -> str:
        response = self.session.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model, "system": SYSTEM_PROMPT,
                "prompt": json.dumps({"question": question, "schema": RetrievalPlan.model_json_schema()}),
                "format": RetrievalPlan.model_json_schema(), "stream": False, "think": False,
            }, timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"]


def _clean_search_query(question: str, spans: list[tuple[int, int]]) -> str:
    chars = list(question)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    words = re.findall(r"[^\W_]+(?:['’][^\W_]+)?", "".join(chars), re.UNICODE)
    stop = {"a", "an", "the", "i", "me", "my", "we", "us", "our", "is", "are", "there", "any",
            "what", "who", "how", "do", "does", "can", "could", "would", "please", "show", "find",
            "looking", "for", "want", "need", "like", "tell", "about", "with", "in", "on", "of", "to",
            "and", "or", "at", "this", "that", "have", "has", "get", "me", "student", "students",
            "research", "degree", "degrees", "project", "projects", "applications", "application",
            "am", "it", "was", "be", "for", "me"}
    meaningful = [word for word in words if word.casefold() not in stop]
    query = " ".join(meaningful).strip(" ?.,!;:")
    return re.sub(r"\bs\s+(?=(?:interests?|background|latest|publications?|h[- ]index|citation|grants?)\b)", "", query, flags=re.I)


def fallback_plan(question: str) -> RetrievalPlan:
    spans, values = [], {}

    def capture(field: str, pattern: str, value_fn=lambda m: m.group(0)):
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            spans.append(match.span())
            values[field] = value_fn(match)

    capture("project_id", r"\b(?:project\s*(?:id\s*)?#?\s*)?(\d{4,8})\b", lambda m: m.group(1))
    capture("degree_type", r"\b(?:master(?:s)?\s+by\s+research|mres)\b", lambda m: "Master by Research")
    if "degree_type" not in values:
        capture("degree_type", r"\bph\s*\.?\s*d\s*\.?\b", lambda m: "PhD")
    capture("student_type", r"\binternational(?:\s+student)?s?\b", lambda m: "International")
    if "student_type" not in values:
        capture("student_type", r"\bdomestic(?:\s+student)?s?\b", lambda m: "Domestic")
    capture("location", r"\b(?:Cradle\s+Coast|Launceston|Hobart|Sydney)\b", lambda m: " ".join(m.group(0).split()).title())
    capture("funding_status", r"\bno\s+stipend\b", lambda m: "no_stipend")
    if "funding_status" not in values:
        capture("funding_status", r"\bfunded\b|\bscholarship\b", lambda m: "funded")
    capture("status", r"\bapplications?\s+open\b|\bopen\s+applications\b", lambda m: "Applications open")
    capture("research_category", r"\b(?:information\s+and\s+communication\s+technology|ICT)\b",
            lambda m: "Information and Communication Technology")
    supervisor_match = re.search(
        r"(?:under|by|from|for)\s+(?:the\s+)?(?:supervisor\s+)?"
        r"((?:(?:associate\s+professor|assoc\.?\s+prof|professor|prof\.?|doctor|dr\.?)\s+)?"
        r"[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3})\b",
        question,
    )
    if supervisor_match is None:
        supervisor_match = re.search(
            r"supervisor\s+(?:(?:named|called)\s+)?"
            r"((?:(?:associate\s+professor|assoc\.?\s+prof|professor|prof\.?|doctor|dr\.?)\s+)?"
            r"[A-Za-z'’-]+(?:\s+[A-Za-z'’-]+){1,3})\b",
            question, re.IGNORECASE,
        )
    if supervisor_match is None:
        supervisor_match = re.search(
            r"(?:does|do)\s+((?:(?:associate\s+professor|assoc\.?\s+prof|professor|prof\.?|doctor|dr\.?)\s+)?"
            r"[A-Za-z'’-]+(?:\s+[A-Za-z'’-]+){1,3})\s+supervis(?:e|es|ing)\b",
            question, re.IGNORECASE,
        )
    if supervisor_match:
        candidate = " ".join(supervisor_match.group(1).split()).strip(" .,;:?")
        # A supervisor query names a person; avoid treating trailing query verbs as part of the name.
        candidate = re.split(r"\s+(?:supervise|supervises|supervising|projects?)\b", candidate,
                             maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if len(candidate.split()) >= 2:
            if not re.search(r"\b(?:researchers?|working|supervised)\b", candidate, re.I):
                values["supervisor"] = candidate
            spans.append(supervisor_match.span(1))
    if "supervisor" not in values:
        profile_match = re.search(
            r"(?:about|of|who\s+is|profile\s+of|research\s+(?:interests?|background)|school\s+is)\s+"
            r"((?:(?:associate\s+professor|assoc\.?\s+prof|professor|prof\.?|doctor|dr\.?)\s+)?"
            r"[A-Za-z][A-Za-z'’-]+(?:\s+[A-Za-z][A-Za-z'’-]+){1,3})",
            question, re.IGNORECASE,
        )
        project_topic = bool(re.search(r"\bprojects?\s+(?:about|on|in|for|related\s+to)\b", question, re.IGNORECASE))
        if profile_match and not project_topic:
            candidate = profile_match.group(1).strip(" .,;:?'\"")
            candidate = re.split(r"\s+(?:latest|research|publication|publications|h-index|works?)\b", candidate, maxsplit=1, flags=re.I)[0].strip()
            candidate = re.split(r"\s+(?:affiliated|working|works?|supervis(?:e|es|ing))\b", candidate, maxsplit=1, flags=re.I)[0].strip()
            if len(candidate.split()) >= 2:
                if not re.search(r"\b(?:researchers?|working|supervised)\b", candidate, re.I):
                    values["supervisor"] = candidate
                spans.append(profile_match.span(1))
    if "supervisor" not in values:
        possessive = re.search(
            r"((?:(?:associate\s+professor|assoc\.?\s+prof|professor|prof\.?|doctor|dr\.?)\s+)?"
            r"[A-Za-z][A-Za-z'’-]+(?:\s+[A-Za-z][A-Za-z'’-]+){1,3})['’]s\s+"
            r"(?:research\s+interests?|research\s+background|school|profile)",
            question, re.IGNORECASE,
        )
        if possessive:
            candidate = re.sub(r"^(?:what\s+are|what\s+is|tell\s+me\s+about)\s+", "", possessive.group(1).strip(), flags=re.I)
            candidate = re.split(r"\s+(?:affiliated|working|works?|supervis(?:e|es|ing))\b", candidate, maxsplit=1, flags=re.I)[0].strip()
            values["supervisor"] = candidate
            spans.append(possessive.span(1))
    if "supervisor" not in values:
        fact_name = re.search(
            r"((?:(?:associate\s+professor|assoc\.?\s+prof|professor|prof\.?|doctor|dr\.?)\s+)?"
            r"[A-Za-z][A-Za-z'’-]+(?:\s+[A-Za-z][A-Za-z'’-]+){1,3})['’]s\s+"
            r"(?:latest\s+)?(?:publications?|h[- ]index|(?:google scholar\s+)?citation count|grants?)",
            question, re.IGNORECASE,
        )
        if fact_name:
            candidate = re.sub(r"^(?:what\s+are|what\s+is|tell\s+me\s+about)\s+", "", fact_name.group(1).strip(), flags=re.I)
            values["supervisor"] = candidate
            spans.append(fact_name.span(1))
    project_signal = bool(values.get("project_id") or re.search(
        r"\b(projects?|opportunities|supervis(?:e|es|or)|find|show\s+me|looking\s+for)\b", question, re.I
    ) or re.search(
        r"\b(?:is|are)\s+there\s+(?:a|any)\s+(?:research\s+)?projects?\s+"
        r"(?:about|on|in|for|related\s+to)\b|\bresearch\s+opportunities\s+(?:in|for|about|on)\b",
        question, re.I,
    ))
    general_signal = bool(re.search(
        r"\b(requirements?|english|apply|application|documents?|fees|scholarships?|how\s+do\s+i)\b",
        question, re.I,
    ))
    local_signal = bool(re.search(r"\b(?:my\s+(?:uploaded|local|private)|local\s+(?:notes?|documents?|files?|preference|rubric|framework|readiness|decision\s+journal)|private\s+(?:notes?|documents?|files?|profile|preference|funding|applicant)|uploaded\s+(?:notes?|documents?|files?)|user[- ]authored|applicant\s+b|persona\s+constraints|decision\s+journal|knowledge\s+base)\b", question, re.I))
    local_signal = local_signal or has_local_document_reference(question)
    scope = "all" if (project_signal and general_signal) or local_signal else "projects" if project_signal else "general"
    query = _clean_search_query(question, spans) or question.strip()
    if values.get("project_id") and not re.search(r"\b(projects?|supervis(?:e|es|or))\b", question, re.I):
        scope = "projects"
    return RetrievalPlan(
        scope=scope, search_query=query, intent=("project_discovery" if scope == "projects" else
                                                 "mixed_research_degree" if scope == "all" else "general_research_degree"),
        confidence=0.45 if values else 0.35, planner_method="fallback", **values,
    )


class QueryPlanner:
    def __init__(self, provider: PlannerProvider | None = None):
        self.provider = provider if provider is not None else OllamaPlanner()
        self.last_llm_error: str | None = None

    @property
    def llm_available(self) -> bool:
        return self.provider.is_available()

    def plan(self, question: str) -> RetrievalPlan:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be blank")
        self.last_llm_error = None
        try:
            if self.provider.is_available():
                payload = json.loads(self.provider.generate(question))
                plan = RetrievalPlan.model_validate(payload)
                return plan.model_copy(update={"planner_method": "llm"})
            self.last_llm_error = "local model unavailable"
        except Exception as exc:
            self.last_llm_error = f"LLM planner failed: {exc}"
        return fallback_plan(question)


REASONING_SYSTEM_PROMPT = """Plan retrieval for the unofficial UTAS research-degree assistant.
Return exactly one JSON object and no prose, markdown, explanations, or reasoning.

Allowed enums:
method = retrieval | graph | hybrid_graph
scope = general | projects | supervisors | all
planner_method = llm
graph_operation = null | project_by_id | projects_by_supervisor | multi_constraint_projects |
  supervisors_with_multiple_projects | supervisors_with_funded_ict_international_projects |
  open_projects_by_category | categories_with_both_degree_types |
  supervisors_across_multiple_categories | count_open_funded_international_ict_projects |
  supervisor_by_name | supervisor_research_fields | supervisors_by_research_field | ict_supervisors_by_research_field |
  ict_supervisors | supervisor_school | supervisors_with_orcid | projects_by_supervisor_research_field

Use retrieval/general for descriptive HDR guidance: applications, entry or English
requirements, scholarships, and required documents. Use retrieval/projects for broad
project discovery without specific structured filters. Use hybrid_graph/projects for
project search/recommendations that have explicit structured constraints. Use graph for
exact lookups, counts, exhaustive relationships, supervisor relationships, category
aggregations, 'more than one', 'across multiple categories', or both degree types.
Use scope=supervisors for supervisor profile descriptions and research-field questions;
use graph operations supervisor_by_name, supervisor_research_fields, supervisor_school,
or supervisors_by_research_field for exact supervisor facts. Use
projects_by_supervisor_research_field for constrained project searches involving a
supervisor research field. Publications, h-index, citation counts, and external grants
are unsupported unless explicitly present in supplied evidence. Graph supervisor plans may
use scope=projects or scope=supervisors. Hybrid_graph plans use scope=projects. An interest keyword is not a
metadata filter: AI does not imply a research_category, and funded does not imply PhD or
Applications open. Set those fields only when the user explicitly states them.

Copy EVERY stated constraint into its field: degree_type, student_type, location,
funding_status, status, research_category, supervisor, project_id. Do not omit one and do
not infer one from research-interest words. Preserve interest words in search_query.
Set unused constraints and graph_operation to null. For graph/hybrid_graph choose exactly
one allowed graph_operation; retrieval must use graph_operation=null. Never write SPARQL.

Output JSON shape:
{"method":"retrieval","scope":"general","search_query":"English score",
 "degree_type":"PhD","student_type":null,"location":null,"funding_status":null,
 "status":null,"research_category":null,"supervisor":null,"project_id":null,
 "graph_operation":null,"intent":"english_requirements","confidence":0.9,
 "planner_method":"llm"}

Output keys/types: method:string, scope:string, search_query:string, degree_type:string|null,
student_type:string|null, location:string|null, funding_status:string|null, status:string|null,
research_category:string|null, supervisor:string|null, project_id:string|null,
graph_operation:string|null, intent:short string label, confidence:number|null,
planner_method:'llm'.

Examples (question followed by its complete JSON output):
A What English score do I need for a PhD?
{"method":"retrieval","scope":"general","search_query":"English score PhD","degree_type":"PhD","student_type":null,"location":null,"funding_status":null,"status":null,"research_category":null,"supervisor":null,"project_id":null,"graph_operation":null,"intent":"english_requirements","confidence":0.9,"planner_method":"llm"}
B Find AI and machine learning PhD projects.
{"method":"retrieval","scope":"projects","search_query":"AI machine learning","degree_type":"PhD","student_type":null,"location":null,"funding_status":null,"status":null,"research_category":null,"supervisor":null,"project_id":null,"graph_operation":null,"intent":"project_discovery","confidence":0.9,"planner_method":"llm"}
C I am an international student looking for funded ICT PhD projects.
{"method":"hybrid_graph","scope":"projects","search_query":"ICT","degree_type":"PhD","student_type":"International","location":null,"funding_status":"funded","status":null,"research_category":"Information and Communication Technology","supervisor":null,"project_id":null,"graph_operation":"multi_constraint_projects","intent":"constrained_project_discovery","confidence":0.9,"planner_method":"llm"}
D Show me funded AI research opportunities in Hobart for international students.
{"method":"hybrid_graph","scope":"projects","search_query":"AI research opportunities","degree_type":null,"student_type":"International","location":"Hobart","funding_status":"funded","status":null,"research_category":null,"supervisor":null,"project_id":null,"graph_operation":"multi_constraint_projects","intent":"constrained_project_discovery","confidence":0.9,"planner_method":"llm"}
E Who supervises project 12259?
{"method":"graph","scope":"projects","search_query":"project supervisor","degree_type":null,"student_type":null,"location":null,"funding_status":null,"status":null,"research_category":null,"supervisor":null,"project_id":"12259","graph_operation":"project_by_id","intent":"project_supervisor_lookup","confidence":0.9,"planner_method":"llm"}
F Which supervisors supervise more than one advertised project?
{"method":"graph","scope":"projects","search_query":"supervisor advertised projects","degree_type":null,"student_type":null,"location":null,"funding_status":null,"status":null,"research_category":null,"supervisor":null,"project_id":null,"graph_operation":"supervisors_with_multiple_projects","intent":"supervisor_project_aggregation","confidence":0.9,"planner_method":"llm"}
G How many open projects are available in each research category?
{"method":"graph","scope":"projects","search_query":"open projects research categories","degree_type":null,"student_type":null,"location":null,"funding_status":null,"status":"Applications open","research_category":null,"supervisor":null,"project_id":null,"graph_operation":"open_projects_by_category","intent":"category_project_counts","confidence":0.9,"planner_method":"llm"}
H Which supervisors have funded ICT projects accepting international students?
{"method":"graph","scope":"projects","search_query":"funded ICT projects international students","degree_type":null,"student_type":"International","location":null,"funding_status":"funded","status":null,"research_category":"Information and Communication Technology","supervisor":null,"project_id":null,"graph_operation":"supervisors_with_funded_ict_international_projects","intent":"funded_ict_supervisors","confidence":0.9,"planner_method":"llm"}
I How many open funded international PhD projects are in ICT?
{"method":"graph","scope":"projects","search_query":"funded international PhD ICT projects","degree_type":"PhD","student_type":"International","location":null,"funding_status":"funded","status":"Applications open","research_category":"Information and Communication Technology","supervisor":null,"project_id":null,"graph_operation":"count_open_funded_international_ict_projects","intent":"constrained_project_count","confidence":0.9,"planner_method":"llm"}
 J Which research categories contain both PhD and Master by Research projects?
 {"method":"graph","scope":"projects","search_query":"PhD Master by Research categories","degree_type":null,"student_type":null,"location":null,"funding_status":null,"status":null,"research_category":null,"supervisor":null,"project_id":null,"graph_operation":"categories_with_both_degree_types","intent":"degree_category_overlap","confidence":0.9,"planner_method":"llm"}
K Tell me about Soonja Yeom.
{"method":"retrieval","scope":"supervisors","search_query":"Soonja Yeom research fields","degree_type":null,"student_type":null,"location":null,"funding_status":null,"status":null,"research_category":null,"supervisor":"Soonja Yeom","project_id":null,"graph_operation":null,"intent":"supervisor_profile_lookup","confidence":0.9,"planner_method":"llm"}
L Which projects are supervised by Soonja Yeom?
{"method":"graph","scope":"projects","search_query":"Soonja Yeom projects","degree_type":null,"student_type":null,"location":null,"funding_status":null,"status":null,"research_category":null,"supervisor":"Soonja Yeom","project_id":null,"graph_operation":"projects_by_supervisor","intent":"supervisor_project_lookup","confidence":0.9,"planner_method":"llm"}
M Which ICT supervisors work in AI?
{"method":"graph","scope":"supervisors","search_query":"Artificial intelligence","degree_type":null,"student_type":null,"location":null,"funding_status":null,"status":null,"research_category":"Artificial intelligence","supervisor":null,"project_id":null,"graph_operation":"supervisors_by_research_field","intent":"supervisor_research_field_lookup","confidence":0.9,"planner_method":"llm"}
N What are Soonja Yeom's latest publications?
{"method":"retrieval","scope":"supervisors","search_query":"Soonja Yeom publications","degree_type":null,"student_type":null,"location":null,"funding_status":null,"status":null,"research_category":null,"supervisor":"Soonja Yeom","project_id":null,"graph_operation":null,"intent":"unsupported_supervisor_profile_fact","confidence":0.9,"planner_method":"llm"}"""


def _reasoning_json_schema() -> dict:
    """Ollama accepts JSON Schema format; require every nullable key to be present."""
    schema = ReasoningPlan.model_json_schema()
    schema["required"] = list(schema["properties"])
    return schema


def _rejected_reasoning_constraints(question: str, plan: ReasoningPlan) -> list[dict]:
    """Return constraints that cannot be tied to explicit wording in the question."""
    explicit = fallback_reasoning_plan(question).project_filters()
    normalized_question = " ".join(question.casefold().split())
    rejected = []
    for field, value in plan.project_filters().items():
        known = explicit.get(field)
        if known is not None and canonical(field, known) == canonical(field, value):
            continue
        normalized_value = " ".join(value.casefold().split())
        if normalized_value not in normalized_question:
            rejected.append({"field": field, "value": value,
                             "reason": "value is not explicit in the user question"})
    return rejected


def _remove_private_reasoning(text: str) -> str:
    """Drop any hidden-thought tags defensively; retain only the final model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def normalize_reasoning_payload(payload: dict) -> dict:
    """Normalize safe aliases/formatting before schema validation; do not infer values."""
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    for key, value in list(normalized.items()):
        if isinstance(value, str):
            normalized[key] = " ".join(value.split())
    for field in ("method", "scope", "planner_method"):
        if isinstance(normalized.get(field), str):
            normalized[field] = normalized[field].casefold()
    aliases = {
        "degree_type": {
            "phd": "PhD", "mres": "Master by Research", "masters by research": "Master by Research",
        },
        "student_type": {
            "international": "International", "domestic": "Domestic",
            "international student": "International", "international students": "International",
            "domestic student": "Domestic", "domestic students": "Domestic",
        },
        "status": {"open": "Applications open", "applications open": "Applications open"},
        "funding_status": {"funded": "funded", "no stipend": "no_stipend"},
        "research_category": {
            "ict": "Information and Communication Technology",
            "ai/ict": "Information and Communication Technology",
        },
    }
    for field, values in aliases.items():
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = values.get(value.casefold(), value)
    return normalized


def _validation_messages(exc: ValidationError) -> list[dict]:
    return [{"location": list(error.get("loc", ())), "message": error.get("msg", "validation error"),
             "type": error.get("type", "validation_error")}
            for error in exc.errors(include_input=False)]


def _failure_categories(reason: str, validation_errors: list[dict] | None = None,
                        rejected_constraints: list[dict] | None = None,
                        missing_constraints: list[str] | None = None) -> list[str]:
    categories = []
    lowered = reason.casefold()
    locations = {str(part) for error in (validation_errors or []) for part in error.get("location", [])}
    if "json" in lowered and ("decode" in lowered or "valid json" in lowered or "json object" in lowered):
        categories.append("invalid JSON")
    if "method" in locations and any("missing" in e.get("type", "") for e in (validation_errors or [])):
        categories.append("missing method")
    if "graph_operation" in locations and any("missing" in e.get("type", "") for e in (validation_errors or [])):
        categories.append("missing graph_operation")
    if "graph_operation" in locations and any("literal_error" in e.get("type", "") for e in (validation_errors or [])):
        categories.append("unsupported graph operation")
    if rejected_constraints:
        categories.append("invented constraint")
    if missing_constraints:
        categories.append("omitted explicit constraint")
    if ("method" in lowered and "scope" in lowered
            or "must use project scope" in lowered
            or "must use project or supervisor scope" in lowered
            or "conflicts with an unambiguous graph-routing rule" in lowered
            or "retrieval method cannot select" in lowered):
        categories.append("incompatible method and scope")
    if "graph_operation" in lowered and "requires" in lowered:
        categories.append("missing graph_operation")
    if not categories and reason:
        categories.append("other")
    return list(dict.fromkeys(categories))


def fallback_reasoning_plan(question: str) -> ReasoningPlan:
    """Small rules for clear cases; ambiguous questions remain retrieval-first."""
    base = fallback_plan(question)
    text = question.casefold()
    values = base.model_dump(exclude={"scope", "search_query", "intent", "confidence", "planner_method"})
    project_id = base.project_id
    has_supervisor = bool(re.search(r"\bsupervisor\w*\b", text))
    has_project = bool(re.search(r"\bprojects?\b", text))
    if values.get("status") is None and has_project and re.search(r"\bopen\b", text):
        values["status"] = "Applications open"
    research_category = bool(re.search(r"\bresearch categor(?:y|ies)\b", text))
    profile_question = bool(re.search(r"\b(tell me about|research interests?|research background|which school|affiliated|profile|who is)\b", text))
    operation = None
    method = "retrieval"
    intent = "general_information"
    scope = base.scope

    local_context = bool(re.search(
        r"\b(?:applicant\s+b|private\s+(?:profile|preference|funding|applicant)|"
        r"local\s+(?:preference|rubric|framework|readiness|decision\s+journal)|"
        r"decision\s+journal|persona\s+constraints|user[- ]authored|"
        r"contact(?:ing)?\s+(?:a\s+)?(?:potential\s+)?supervisor|"
        r"publication\s+information.*knowledge\s+base)\b", text, re.I))
    if local_context or has_local_document_reference(question):
        return ReasoningPlan(
            method="retrieval", scope="all", search_query=question.strip(),
            intent="local_document_lookup", confidence=max(base.confidence or 0, 0.55),
            planner_method="fallback",
        )

    if project_id and has_supervisor and re.search(r"\b(who supervis|supervisor)\b", text):
        method, scope, operation, intent = "graph", "projects", "project_by_id", "project_supervisor_lookup"
    elif base.supervisor and project_id and re.search(r"\b(research interests?|background|profile|school)\b", text):
        method, scope, operation, intent = "graph", "projects", "project_by_id", "project_supervisor_profile_lookup"
    elif base.supervisor and re.search(r"\b(?:latest\s+publications?|publications?|h-index|citation count|grants?)\b", text):
        method, scope, operation, intent = "retrieval", "supervisors", None, "unsupported_supervisor_profile_fact"
    elif re.search(r"\bsupervisors?\b", text) and re.search(r"\borcid\b", text):
        method, scope, operation, intent = "graph", "supervisors", "supervisors_with_orcid", "supervisor_metadata_lookup"
    elif base.supervisor and re.search(r"\b(?:which\s+school|affiliated)\b", text):
        method, scope, operation, intent = "graph", "supervisors", "supervisor_school", "supervisor_school_lookup"
    elif base.supervisor and re.search(r"\b(?:tell me about|research interests?|research background|which school|affiliated|who is)\b", text):
        method, scope, operation, intent = "retrieval", "supervisors", None, "supervisor_profile_lookup"
    elif ("ict" in text and re.search(r"projects?\s+supervised|researchers?\s+working", text)):
        method, scope, operation, intent = "hybrid_graph", "projects", "projects_by_supervisor_research_field", "project_supervisor_field_discovery"
        values["research_category"] = "Information and Communication Technology"
    elif re.search(r"\b(?:supervisors?|researchers?)\b", text) and re.search(r"\b(?:ai|artificial intelligence)\b", text):
        operation = "ict_supervisors_by_research_field" if "ict" in text else "supervisors_by_research_field"
        method, scope, operation, intent = "graph", "supervisors", operation, "supervisor_research_field_lookup"
        values["research_category"] = "Artificial intelligence"
    elif base.supervisor and re.search(r"\b(?:projects?|supervis)\b", text):
        method, scope, operation, intent = "graph", "projects", "projects_by_supervisor", "supervisor_project_lookup"
    elif (has_supervisor and "funded" in text and "ict" in text
          and re.search(r"\binternational\b", text)):
        method, scope, operation, intent = (
            "graph", "projects", "supervisors_with_funded_ict_international_projects", "supervisor_project_relationship"
        )
    elif (has_supervisor and research_category and
          re.search(r"\b(across|more than one|multiple|span|work across)\b", text)):
        method, scope, operation, intent = (
            "graph", "projects", "supervisors_across_multiple_categories", "supervisor_category_aggregation"
        )
    elif (research_category and re.search(r"\b(how many|count|number)\b", text)
          and "open" in text):
        method, scope, operation, intent = "graph", "projects", "open_projects_by_category", "category_project_counts"
    elif (research_category and "phd" in text and
          re.search(r"\b(master(?:s)? by research|mres)\b", text)):
        method, scope, operation, intent = "graph", "projects", "categories_with_both_degree_types", "degree_category_overlap"
        # These degree names define the comparison, rather than a single project filter.
        values["degree_type"] = None
    elif (has_supervisor and re.search(r"\b(more than one|multiple|at least two)\b", text)
          and has_project):
        method, scope, operation, intent = (
            "graph", "projects", "supervisors_with_multiple_projects", "supervisor_project_aggregation"
        )
    elif (re.search(r"\b(how many|count|number)\b", text) and "open" in text and "funded" in text
          and "international" in text and "phd" in text and "ict" in text):
        method, scope, operation, intent = (
            "graph", "projects", "count_open_funded_international_ict_projects", "constrained_project_count"
        )
    elif project_id:
        method, scope, operation, intent = "graph", "projects", "project_by_id", "project_lookup"
    elif base.supervisor:
        method, scope, operation, intent = "graph", "projects", "projects_by_supervisor", "supervisor_project_lookup"
    elif base.scope == "projects":
        intent = "project_discovery"
        structured = any(getattr(base, field) for field in (
            "student_type", "location", "funding_status", "status", "research_category", "supervisor"
        ))
        if structured:
            method, operation = "hybrid_graph", "multi_constraint_projects"
            intent = "constrained_project_discovery"

    search_query = base.search_query
    if scope == "supervisors" and base.supervisor:
        search_query = f"{base.supervisor} research fields"
    if operation == "multi_constraint_projects":
        scope = "projects"
    return ReasoningPlan(
        method=method, scope=scope, search_query=search_query,
        graph_operation=operation, intent=intent,
        confidence=0.45 if operation or base.project_filters() else 0.35,
        planner_method="fallback", **values,
    )


class OllamaReasoningPlanner:
    """Ollama provider specialized for the controlled reasoning-plan schema."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL,
                 timeout: float = OLLAMA_TIMEOUT_SECONDS, session=None):
        self.host, self.model, self.timeout = host.rstrip("/"), model, timeout
        self.session = session or requests.Session()
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=min(self.timeout, 2))
            response.raise_for_status()
            names = {entry.get("name") for entry in response.json().get("models", [])}
            self._available = self.model in names or any(
                name and name.split(":")[0] == self.model.split(":")[0] for name in names
            )
        except (requests.RequestException, ValueError, AttributeError):
            self._available = False
        return self._available

    def generate(self, question: str) -> str:
        response = self.session.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model, "system": REASONING_SYSTEM_PROMPT,
                "prompt": json.dumps({"question": question, "schema": _reasoning_json_schema()}),
                "format": _reasoning_json_schema(), "stream": False, "think": False,
            }, timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"]


class ReasoningPlanner:
    """Primary local-LLM planner with a minimal deterministic fallback."""

    def __init__(self, provider: PlannerProvider | None = None):
        self.provider = provider if provider is not None else OllamaReasoningPlanner()
        self.last_llm_error: str | None = None
        self.last_diagnostics: dict = {}

    @property
    def llm_available(self) -> bool:
        return self.provider.is_available()

    def plan(self, question: str) -> ReasoningPlan:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be blank")
        self.last_llm_error = None
        expected = fallback_reasoning_plan(question)
        diagnostics = {
            "question": question,
            "raw_model_response": None,
            "parsed_json": None,
            "normalized_json": None,
            "validation_errors": [],
            "rejected_constraints": [],
            "missing_explicit_constraints": [],
            "reason_for_fallback": None,
            "expected_broad_routing_category": {
                "method": expected.method, "scope": expected.scope,
                "graph_operation": expected.graph_operation,
            },
            "failure_categories": [],
            "qwen_plan_accepted": False,
        }
        self.last_diagnostics = diagnostics
        try:
            if not self.provider.is_available():
                raise RuntimeError("local model unavailable")
            raw = _remove_private_reasoning(self.provider.generate(question))
            diagnostics["raw_model_response"] = raw
            payload = json.loads(raw)  # Strict JSON-only parsing; no markdown/prose repair.
            diagnostics["parsed_json"] = payload
            normalized_payload = normalize_reasoning_payload(payload)
            diagnostics["normalized_json"] = normalized_payload
            try:
                plan = ReasoningPlan.model_validate(normalized_payload)
            except ValidationError as exc:
                diagnostics["validation_errors"] = _validation_messages(exc)
                raise

            rejected = _rejected_reasoning_constraints(question, plan)
            diagnostics["rejected_constraints"] = rejected
            if rejected:
                raise ValueError("LLM planner emitted constraint values not explicit in the question")
            explicit_plan = fallback_reasoning_plan(question)
            if explicit_plan.intent == "local_document_lookup" and (
                plan.method != "retrieval" or plan.scope != "all"
                or plan.search_query != question or plan.project_filters()
                or plan.graph_operation is not None
                or plan.intent != explicit_plan.intent
            ):
                raise ValueError("LLM plan changed explicit local-document routing or original search query")
            for field, value in explicit_plan.project_filters().items():
                planned_value = plan.project_filters().get(field)
                if planned_value is None:
                    diagnostics["missing_explicit_constraints"].append(field)
                    raise ValueError(f"LLM planner omitted explicit {field} constraint")
                if canonical(field, planned_value) != canonical(field, value):
                    diagnostics["missing_explicit_constraints"].append(field)
                    raise ValueError(f"LLM planner changed explicit {field} constraint")
            if explicit_plan.method in {"graph", "hybrid_graph"} and (
                plan.method != explicit_plan.method or plan.graph_operation != explicit_plan.graph_operation
            ):
                raise ValueError("LLM plan conflicts with an unambiguous graph-routing rule")
            if explicit_plan.scope == "projects" and plan.scope == "general":
                raise ValueError("LLM plan omitted clear project-discovery scope")
            if plan.method in {"graph", "hybrid_graph"} and not plan.graph_operation:
                raise ValueError("graph method requires a controlled graph_operation")
            if plan.method == "retrieval" and plan.graph_operation:
                raise ValueError("retrieval method cannot select a graph_operation")
            if plan.method == "graph" and plan.scope not in {"projects", "supervisors"}:
                raise ValueError("graph must use project or supervisor scope")
            if plan.method == "hybrid_graph" and plan.scope != "projects":
                raise ValueError("hybrid_graph must use project scope")
            if plan.method == "hybrid_graph" and not any(
                plan.project_filters().get(field) for field in (
                    "student_type", "location", "funding_status", "status", "research_category",
                    "supervisor", "project_id",
                )
            ):
                raise ValueError("hybrid_graph requires at least one explicit project constraint")
            diagnostics["qwen_plan_accepted"] = True
            return plan.model_copy(update={"planner_method": "llm"})
        except json.JSONDecodeError as exc:
            diagnostics["validation_errors"].append({"type": "json_decode_error", "message": str(exc)})
            reason = f"invalid JSON: {exc}"
            diagnostics["failure_categories"] = ["invalid JSON"]
            self.last_llm_error = reason
        except ValidationError as exc:
            if not diagnostics["validation_errors"]:
                diagnostics["validation_errors"] = _validation_messages(exc)
            reason = f"ReasoningPlan validation failed: {exc}"
            diagnostics["failure_categories"] = _failure_categories(
                reason, diagnostics["validation_errors"], diagnostics["rejected_constraints"],
                diagnostics["missing_explicit_constraints"],
            )
            self.last_llm_error = reason
        except Exception as exc:
            reason = str(exc)
            if not reason:
                reason = exc.__class__.__name__
            if diagnostics["raw_model_response"] is not None and not diagnostics["validation_errors"]:
                diagnostics["validation_errors"].append({"type": "planner_guard", "message": reason})
            diagnostics["failure_categories"] = _failure_categories(
                reason, diagnostics["validation_errors"], diagnostics["rejected_constraints"],
                diagnostics["missing_explicit_constraints"],
            )
            self.last_llm_error = f"LLM reasoning planner failed: {reason}"
        diagnostics["reason_for_fallback"] = self.last_llm_error
        return fallback_reasoning_plan(question)
