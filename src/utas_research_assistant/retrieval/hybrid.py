"""BM25 and semantic rank fusion using Reciprocal Rank Fusion."""

from dataclasses import dataclass

from utas_research_assistant.retrieval.bm25 import BM25Retriever
from utas_research_assistant.retrieval.corpus import Corpus
from utas_research_assistant.retrieval.filters import filter_projects
from utas_research_assistant.retrieval.local_documents import resolve_local_document
from utas_research_assistant.retrieval.project_documents import ProjectDocument
from utas_research_assistant.retrieval.semantic import SemanticRetriever

ALLOWED_SCOPES = {"all", "projects", "general", "supervisors"}


def result_key(result: dict) -> tuple:
    provenance = result["provenance"][0]
    identity = provenance.get("chunk_id") or result.get("project_id") or provenance.get("document_id")
    return result["item_type"], identity


def rrf_score(*, bm25_rank: int | None, semantic_rank: int | None, k: int = 60) -> float:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    return sum(1 / (k + rank) for rank in (bm25_rank, semantic_rank) if rank is not None)


@dataclass
class HybridRetriever:
    corpus: Corpus
    bm25: BM25Retriever
    semantic: SemanticRetriever
    rrf_k: int = 60

    def __post_init__(self):
        if isinstance(self.rrf_k, bool) or not isinstance(self.rrf_k, int) or self.rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer")

    def candidate_indices(self, scope: str = "all", filters: dict | None = None,
                          candidate_project_ids: set[str] | None = None) -> list[int]:
        if scope not in ALLOWED_SCOPES:
            raise ValueError(f"scope must be one of: {', '.join(sorted(ALLOWED_SCOPES))}")
        if scope == "general" and filters:
            raise ValueError("Project filters cannot be applied to general-document scope")
        scope_type = {"projects": "research_project", "general": "general_chunk", "supervisors": "supervisor_profile"}.get(scope)
        indices = [i for i, item in enumerate(self.corpus.items)
                   if scope == "all" or item.item_type == scope_type]
        if scope == "supervisors" and filters:
            raise ValueError("Project filters cannot be applied to supervisor scope")
        if filters and scope in {"all", "projects"}:
            project_indices = [i for i in indices if self.corpus.items[i].item_type == "research_project"]
            projects = [ProjectDocument.model_validate({
                **self.corpus.items[i].metadata, "text": self.corpus.items[i].text,
            }) for i in project_indices]
            eligible_ids = {project.project_id for project in filter_projects(projects, **filters)}
            eligible = {i for i in project_indices if self.corpus.items[i].metadata["project_id"] in eligible_ids}
            # Explicit project constraints do not suppress general guidance in all scope.
            indices = [i for i in indices if self.corpus.items[i].item_type == "general_chunk" or i in eligible]
        if candidate_project_ids is not None:
            candidate_project_ids = {str(project_id) for project_id in candidate_project_ids}
            indices = [i for i in indices if self.corpus.items[i].item_type == "general_chunk"
                       or str(self.corpus.items[i].metadata.get("project_id")) in candidate_project_ids]
        return indices

    def search(self, query: str, top_k: int = 5, scope: str = "all",
               filters: dict | None = None, *,
               candidate_project_ids: set[str] | None = None) -> list[dict]:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")
        if top_k <= 0:
            return []
        candidates = self.candidate_indices(scope, filters, candidate_project_ids)
        target = resolve_local_document(query, (self.corpus.items[i] for i in candidates))
        if target is not None:
            candidates = [i for i in candidates
                          if self.corpus.items[i].item_type == "local_document"
                          and self.corpus.items[i].metadata.get("document_id") == target]
        if not candidates:
            return []
        selected = [self.corpus.items[i] for i in candidates]
        scoped_corpus = Corpus(selected, len(selected))
        bm25 = BM25Retriever(scoped_corpus).search(query, len(selected))
        semantic = self.semantic.search(query, len(selected), candidate_indices=set(candidates))
        results = {}
        for rank, result in enumerate(bm25, 1):
            key = result_key(result)
            entry = results.setdefault(key, {"bm25_rank": None, "bm25_score": None,
                                             "semantic_rank": None, "semantic_score": None})
            entry.update(bm25_rank=rank, bm25_score=result["score"])
            entry["record"] = result
        for rank, result in enumerate(semantic, 1):
            key = result_key(result)
            entry = results.setdefault(key, {"bm25_rank": None, "bm25_score": None,
                                             "semantic_rank": None, "semantic_score": None})
            entry.update(semantic_rank=rank, semantic_score=result["score"])
            if "record" not in entry:
                entry["record"] = result

        ordered = []
        for key, entry in results.items():
            score = rrf_score(bm25_rank=entry["bm25_rank"],
                              semantic_rank=entry["semantic_rank"], k=self.rrf_k)
            ordered.append((key, entry, score))
        ordered.sort(key=lambda item: (-item[2], item[0]))
        output = []
        for rank, (_, entry, score) in enumerate(ordered[:top_k], 1):
            record = entry["record"]
            output.append({
                "rank": rank, "score": score, "rrf_score": score,
                "bm25_rank": entry["bm25_rank"], "bm25_score": entry["bm25_score"],
                "semantic_rank": entry["semantic_rank"], "semantic_score": entry["semantic_score"],
                **{field: record.get(field) for field in (
                "item_type", "title", "text", "source_url", "project_id",
                    "primary_supervisor", "research_categories", "category", "provenance",
                    "document_id", "chunk_id",
                    "supervisor_id", "canonical_name", "discovery_profile_id", "discovery_url",
                    "school", "research_fields", "related_project_ids", "related_research_categories",
                    "is_ict_supervisor", "orcid", "google_scholar_url",
                    "supervisor_profile",
                    "local_filename", "source_path", "file_type",
                )},
                "degree_types": record.get("degree_types") or record["provenance"][0].get("degree_types"),
                "student_types": record.get("student_types") or record["provenance"][0].get("student_types"),
                "location": record.get("location", record["provenance"][0].get("location")),
                "funding_status": record.get("funding_status", record["provenance"][0].get("funding_status")),
                "status": record.get("status", record["provenance"][0].get("status")),
                "matched_passage": record.get("matched_passage"),
            })
        return output
