"""Cached CPU sentence embeddings over the same corpus used by BM25."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from utas_research_assistant.retrieval.corpus import Corpus, tokenize
from utas_research_assistant.retrieval.semantic_passages import embedding_items

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def configured_model() -> str:
    return os.environ.get("UTAS_EMBEDDING_MODEL", DEFAULT_MODEL)


def load_model(name: str, *, allow_download: bool = False):
    # Lazy import keeps mocked tests independent of torch and model downloads.
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(name, device="cpu", local_files_only=True)
    except OSError:
        if not allow_download:
            raise
        return SentenceTransformer(name, device="cpu", local_files_only=False)


def corpus_rows(corpus: Corpus) -> list[dict]:
    return [asdict(item) for item in corpus.items]


def fingerprint(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(vectors) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Embedding model returned a zero vector")
    return vectors / norms


def build_embeddings(corpus: Corpus, directory: Path, *, model_name: str = DEFAULT_MODEL,
                     batch_size: int = 32, model=None) -> dict:
    if not corpus.items or batch_size <= 0:
        raise ValueError("A nonempty corpus and positive batch_size are required")
    start = perf_counter()
    encoder = model if model is not None else load_model(model_name, allow_download=True)
    passages = embedding_items(corpus, encoder)
    vectors = normalize(encoder.encode(
        [item["text"] for item in passages], batch_size=batch_size,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
    ))
    if vectors.shape[0] != len(passages) or vectors.shape[1] == 0:
        raise ValueError("Encoder output does not match corpus item count")
    rows = corpus_rows(corpus)
    general = [p for p in passages if p["item_type"] == "general_chunk"]
    sizes = [p["word_count"] for p in general]
    metadata = {
        "format_version": 2,
        "model_name": model_name, "embedding_dimension": vectors.shape[1],
        "corpus_item_count": len(rows), "built_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": round(perf_counter() - start, 3), "device": "cpu",
        "normalized": True, "batch_size": batch_size,
        "max_sequence_length": getattr(encoder, "max_seq_length", None),
        "corpus": corpus.summary, "corpus_fingerprint": fingerprint(rows), "items": rows,
        "embedding_item_count": len(passages), "embedding_items": passages,
        "embedding_items_fingerprint": fingerprint(passages),
        "passage_summary": {
            "general_passage_count": len(general), "project_embedding_count": len(passages) - len(general),
            "min_words": min(sizes, default=0), "average_words": round(sum(sizes) / len(sizes), 2) if sizes else 0,
            "max_words": max(sizes, default=0), "above_240_words": sum(n > 240 for n in sizes),
            "max_model_tokens": max(p["token_count"] for p in passages),
            "above_model_limit": sum(p["token_count"] > encoder.max_seq_length for p in passages),
        },
        "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
    }
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "semantic_embeddings.npy", vectors, allow_pickle=False)
    (directory / "semantic_passages.json").write_text(
        json.dumps(general, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / "semantic_index.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


class SemanticRetriever:
    def __init__(self, corpus: Corpus, directory: Path, *, model=None):
        self.corpus = corpus
        self.metadata = json.loads((directory / "semantic_index.json").read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != 2:
            raise ValueError("Old semantic index format; run build_embeddings.py")
        self.vectors = np.load(directory / "semantic_embeddings.npy", allow_pickle=False)
        rows = corpus_rows(corpus)
        if (self.metadata["items"] != rows
                or self.metadata["corpus_fingerprint"] != fingerprint(rows)):
            raise ValueError("Semantic index is stale or reordered; run build_embeddings.py")
        self.passages = self.metadata["embedding_items"]
        if fingerprint(self.passages) != self.metadata["embedding_items_fingerprint"]:
            raise ValueError("Embedding passage mapping mismatch; rebuild the index")
        for passage in self.passages:
            parent = passage["parent_item_index"]
            if not isinstance(parent, int) or not 0 <= parent < len(rows):
                raise ValueError("Invalid embedding parent mapping")
            item = corpus.items[parent]
            if (passage["parent_chunk_id"] != item.metadata.get("chunk_id")
                    or passage["document_id"] != item.metadata["document_id"]
                    or passage["provenance"] != item.provenance
                    or passage["text"].split() != item.text.split()[passage["start_word"]:passage["end_word"]]):
                raise ValueError("Embedding passage does not match parent corpus item")
        expected = (len(self.passages), self.metadata["embedding_dimension"])
        if (self.metadata["corpus_item_count"] != len(rows) or self.vectors.shape != expected
                or self.metadata["embedding_item_count"] != len(self.passages)
                or self.metadata["vectors_sha256"] != hashlib.sha256(self.vectors.tobytes()).hexdigest()
                or not np.isfinite(self.vectors).all()
                or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5)):
            raise ValueError("Embedding/index mismatch or invalid vectors; rebuild the index")
        self.model = model

    def search(self, query: str, top_k: int = 5, *, candidate_indices: set[int] | None = None) -> list[dict]:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")
        if top_k <= 0 or not tokenize(query) or not self.corpus.items:
            return []
        if self.model is None:
            self.model = load_model(self.metadata["model_name"])
        query_vector = normalize(self.model.encode(
            [query], batch_size=1, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ))
        if query_vector.shape != (1, self.vectors.shape[1]):
            raise ValueError("Query model dimension does not match cached embeddings")
        scores = self.vectors @ query_vector[0]
        # Rank each original item by its best passage; never return duplicate parents.
        best = {}
        for index, passage in enumerate(self.passages):
            parent = passage["parent_item_index"]
            if candidate_indices is not None and parent not in candidate_indices:
                continue
            if parent not in best or scores[index] > scores[best[parent]]:
                best[parent] = index
        order = sorted(best, key=lambda parent: (-float(scores[best[parent]]), parent))[:top_k]
        results = []
        for rank, index in enumerate(order, 1):
            item = self.corpus.items[index]
            metadata = item.metadata
            results.append({
                "rank": rank, "score": float(scores[best[index]]), "item_type": item.item_type,
                "title": metadata["title"], "text": item.text,
                "source_url": metadata.get("source_url"), "project_id": metadata.get("project_id"),
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
                "matched_passage": self.passages[best[index]],
            })
        return results
