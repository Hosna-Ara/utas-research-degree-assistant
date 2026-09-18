import json

import numpy as np
import pytest

from utas_research_assistant.retrieval.corpus import Corpus, CorpusItem
from utas_research_assistant.retrieval import semantic


class FakeEncoder:
    max_seq_length = 256

    def __init__(self):
        self.calls = []

    def tokenizer(self, text, **kwargs):
        return {"input_ids": [0] + list(range(len(text.split()))) + [1]}

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        vectors = {"automobile": [2, 0], "ocean": [0, 3], "transport": [3, 4], "car": [4, 0]}
        return np.array([vectors[text] for text in texts], dtype=np.float32)


@pytest.fixture(autouse=True)
def forbid_model_download(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unit tests must never load or download a real model")
    monkeypatch.setattr(semantic, "load_model", forbidden)


@pytest.fixture
def corpus():
    items = []
    for i, text in enumerate(("automobile", "ocean", "transport")):
        metadata = {
            "title": text, "source_url": "https://www.utas.edu.au/",
            "project_id": str(i), "primary_supervisor": "Doctor Example",
            "research_categories": ["Engineering"], "document_id": f"doc-{i}",
        }
        items.append(CorpusItem("research_project", text, [text], metadata, [dict(metadata)]))
    return Corpus(items, 3)


def test_build_cache_metadata_and_normalization(tmp_path, corpus):
    encoder = FakeEncoder()
    metadata = semantic.build_embeddings(corpus, tmp_path, model_name="mock", batch_size=2, model=encoder)
    assert metadata["model_name"] == "mock"
    assert metadata["embedding_dimension"] == 2
    assert metadata["corpus_item_count"] == 3
    assert metadata["built_at"]
    assert metadata["build_seconds"] >= 0
    assert metadata["items"] == semantic.corpus_rows(corpus)
    assert encoder.calls[0][0] == [item.text for item in corpus.items]
    assert encoder.calls[0][1]["batch_size"] == 2
    assert encoder.calls[0][1]["normalize_embeddings"] is True
    vectors = np.load(tmp_path / "semantic_embeddings.npy")
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)
    np.testing.assert_allclose(vectors, [[1, 0], [0, 1], [.6, .8]])


def test_ranking_metadata_and_cached_document_vectors(tmp_path, corpus):
    encoder = FakeEncoder()
    semantic.build_embeddings(corpus, tmp_path, model_name="mock", model=encoder)
    retriever = semantic.SemanticRetriever(corpus, tmp_path, model=encoder)
    results = retriever.search("car")
    assert [r["title"] for r in results] == ["automobile", "transport", "ocean"]
    assert [r["rank"] for r in results] == [1, 2, 3]
    np.testing.assert_allclose([r["score"] for r in results], [1, .6, 0])
    first = results[0]
    for key in ("title", "source_url", "project_id", "primary_supervisor", "research_categories"):
        assert first[key] == corpus.items[0].metadata[key]
    assert first["provenance"] == corpus.items[0].provenance
    assert first["text"] == "automobile"
    assert encoder.calls[-1][0] == ["car"]  # Documents aren't re-encoded on search.
    assert retriever.search("car") == results


def test_empty_queries_and_top_k(tmp_path, corpus):
    encoder = FakeEncoder()
    semantic.build_embeddings(corpus, tmp_path, model_name="mock", model=encoder)
    retriever = semantic.SemanticRetriever(corpus, tmp_path, model=encoder)
    for query in ("", "  ", "?!"):
        assert retriever.search(query) == []
    assert len(encoder.calls) == 1
    assert retriever.search("car", top_k=0) == []
    assert retriever.search("car", top_k=-1) == []
    assert len(retriever.search("car", top_k=1)) == 1
    assert len(retriever.search("car", top_k=100)) == 3
    for invalid in (True, 1.5):
        with pytest.raises(ValueError, match="integer"):
            retriever.search("car", top_k=invalid)


def test_reordered_or_changed_corpus_rejected(tmp_path, corpus):
    semantic.build_embeddings(corpus, tmp_path, model_name="mock", model=FakeEncoder())
    with pytest.raises(ValueError, match="stale or reordered"):
        semantic.SemanticRetriever(Corpus(list(reversed(corpus.items)), 3), tmp_path)
    corpus.items[0].metadata["title"] = "Changed title"
    with pytest.raises(ValueError, match="stale or reordered"):
        semantic.SemanticRetriever(corpus, tmp_path)


@pytest.mark.parametrize("corruption", ["dimension", "count", "vector_order"])
def test_index_vector_consistency(tmp_path, corpus, corruption):
    semantic.build_embeddings(corpus, tmp_path, model_name="mock", model=FakeEncoder())
    path = tmp_path / "semantic_index.json"
    metadata = json.loads(path.read_text())
    if corruption == "dimension":
        metadata["embedding_dimension"] = 99
    elif corruption == "count":
        metadata["corpus_item_count"] = 99
    else:
        vectors = np.load(tmp_path / "semantic_embeddings.npy")
        np.save(tmp_path / "semantic_embeddings.npy", vectors[::-1])
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mismatch"):
        semantic.SemanticRetriever(corpus, tmp_path)


def test_general_metadata_and_deterministic_alignment(tmp_path, corpus):
    corpus.items[0].item_type = "general_chunk"
    corpus.items[0].metadata["category"] = "faq"
    corpus.items[0].provenance.append({"document_id": "other-source", "category": "entry_requirements"})
    first = semantic.build_embeddings(corpus, tmp_path, model_name="mock", model=FakeEncoder())
    vectors = np.load(tmp_path / "semantic_embeddings.npy").copy()
    second = semantic.build_embeddings(corpus, tmp_path, model_name="mock", model=FakeEncoder())
    np.testing.assert_array_equal(vectors, np.load(tmp_path / "semantic_embeddings.npy"))
    assert first["corpus_fingerprint"] == second["corpus_fingerprint"]
    result = semantic.SemanticRetriever(corpus, tmp_path, model=FakeEncoder()).search("car", 1)[0]
    assert result["item_type"] == "general_chunk"
    assert result["category"] == "faq"
    assert len(result["provenance"]) == 2


def test_invalid_embeddings_rejected(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        semantic.build_embeddings(Corpus([], 0), tmp_path, model=FakeEncoder())
    with pytest.raises(ValueError, match="zero vector"):
        semantic.normalize([[0, 0]])
    with pytest.raises(ValueError, match="finite"):
        semantic.normalize([[np.nan, 1]])


def test_hybrid_with_successfully_loaded_model(tmp_path, corpus, monkeypatch):
    from utas_research_assistant.retrieval.hybrid import HybridRetriever

    encoder = FakeEncoder()
    semantic.build_embeddings(corpus, tmp_path, model_name='mock', model=encoder)
    loads = []

    def cached_model(name):
        loads.append(name)
        return encoder

    monkeypatch.setattr(semantic, 'load_model', cached_model)
    retriever = semantic.SemanticRetriever(corpus, tmp_path)
    hybrid = HybridRetriever(corpus, None, retriever)
    for _ in range(2):
        rows = hybrid.search('automobile')
        assert rows[0]['title'] == 'automobile'
        assert rows[0]['bm25_rank'] == 1
        assert rows[0]['semantic_rank'] == 1
        assert rows[0]['rrf_score'] == pytest.approx(2 / 61)
    assert loads == ['mock']
