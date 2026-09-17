import json

import numpy as np
import pytest

from utas_research_assistant.retrieval.corpus import Corpus, CorpusItem
from utas_research_assistant.retrieval.semantic import build_embeddings, SemanticRetriever
from utas_research_assistant.retrieval.semantic_passages import passage_spans


class Encoder:
    max_seq_length = 256

    def tokenizer(self, text, **kwargs):
        return {"input_ids": [0] * (len(text.split()) * 2 + 2)}

    def encode(self, texts, **kwargs):
        return np.array([[1, 0] if "answer" in t else [0, 1] for t in texts], dtype=np.float32)


def test_word_targets_and_paragraph_boundaries():
    text = "\n\n".join(" ".join(f"p{p}w{i}" for i in range(100)) for p in range(6))
    tokenizer = lambda t, **kw: {"input_ids": [0] * (len(t.split()) + 2)}
    spans = list(passage_spans(text, tokenizer, 256))
    assert spans[0][1] == 200
    assert all(180 <= end - start <= 240 for start, end, _ in spans[:-1])
    assert spans[-1][1] == 600
    for left, right in zip(spans, spans[1:]):
        assert left[1] - right[0] == 40
    words = spans[0][2].split()
    for _, _, value in spans[1:]:
        words.extend(value.split()[40:])
    assert words == text.split()


def test_token_limit_overrides_word_target():
    text = " ".join(f"word{i}" for i in range(500))
    encoder = Encoder()
    spans = list(passage_spans(text, encoder.tokenizer, encoder.max_seq_length))
    assert all(len(encoder.tokenizer(value)["input_ids"]) <= 256 for _, _, value in spans)
    assert max(end - start for start, end, _ in spans) == 127


def test_parent_mapping_alignment_and_unique_results(tmp_path):
    text = " ".join(["ordinary"] * 200 + ["answer"] + ["ordinary"] * 200)
    metadata = {"chunk_id": "chunk1", "document_id": "doc1", "title": "Title", "category": "faq",
                "source_url": "https://www.utas.edu.au/", "local_filename": "faq.html", "page_number": 2}
    corpus = Corpus([CorpusItem("general_chunk", text, text.split(), metadata, [metadata])], 1)
    report = build_embeddings(corpus, tmp_path, model=Encoder(), model_name="fake")
    assert report["embedding_item_count"] > report["corpus_item_count"]
    assert report["passage_summary"]["above_model_limit"] == 0
    retriever = SemanticRetriever(corpus, tmp_path, model=Encoder())
    result, = retriever.search("answer", top_k=5)
    assert result["score"] == 1
    assert result["text"] == text
    assert result["provenance"] == [metadata]
    assert result["matched_passage"]["parent_chunk_id"] == "chunk1"
    assert "answer" in result["matched_passage"]["text"]
    assert result["matched_passage"]["page_number"] == 2
    previous_ids = [p["passage_id"] for p in report["embedding_items"]]
    again = build_embeddings(corpus, tmp_path, model=Encoder(), model_name="fake")
    assert previous_ids == [p["passage_id"] for p in again["embedding_items"]]
    path = tmp_path / "semantic_index.json"
    damaged = json.loads(path.read_text())
    damaged["embedding_items"][0]["parent_item_index"] = 9
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="mapping mismatch"):
        SemanticRetriever(corpus, tmp_path, model=Encoder())
