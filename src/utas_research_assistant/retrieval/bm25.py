"""Deterministic BM25Okapi baseline, without field boosts or query expansion."""

from rank_bm25 import BM25Okapi

from utas_research_assistant.retrieval.corpus import Corpus, tokenize


class BM25Retriever:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.index = BM25Okapi([item.tokens for item in corpus.items]) if corpus.items else None
        self.vocabularies = [set(item.tokens) for item in corpus.items]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")
        query_tokens = tokenize(query)
        if top_k <= 0 or not query_tokens or self.index is None:
            return []
        scores = self.index.get_scores(query_tokens)
        query_words = set(query_tokens)
        # Exclude no-match documents, but keep matching documents even with zero/negative IDF.
        matches = [i for i, words in enumerate(self.vocabularies) if words & query_words]
        matches.sort(key=lambda i: (-float(scores[i]), i))
        results = []
        for rank, index in enumerate(matches[:top_k], 1):
            item = self.corpus.items[index]
            metadata = item.metadata
            results.append({
                "rank": rank, "score": float(scores[index]), "item_type": item.item_type,
                "title": metadata["title"], "text": item.text,
                "source_url": metadata.get("source_url"),
                "project_id": metadata.get("project_id"),
                "primary_supervisor": metadata.get("primary_supervisor"),
                "research_categories": metadata.get("research_categories"),
                "category": metadata.get("category"), "provenance": item.provenance,
                "document_id": metadata.get("document_id"), "chunk_id": metadata.get("chunk_id"),
                "supervisor_id": metadata.get("supervisor_id"),
                "canonical_name": metadata.get("canonical_name"),
                "discovery_profile_id": metadata.get("discovery_profile_id"),
                "discovery_url": metadata.get("discovery_url"),
                "school": metadata.get("school"),
                "research_fields": metadata.get("research_fields"),
                "orcid": metadata.get("orcid"),
                "google_scholar_url": metadata.get("google_scholar_url"),
                "related_project_ids": metadata.get("related_project_ids"),
                "related_research_categories": metadata.get("related_research_categories"),
                "is_ict_supervisor": metadata.get("is_ict_supervisor"),
                "supervisor_profile": metadata.get("supervisor_profile"),
                "local_filename": metadata.get("local_filename"),
                "source_path": metadata.get("source_path"),
                "file_type": metadata.get("file_type"),
            })
        return results
