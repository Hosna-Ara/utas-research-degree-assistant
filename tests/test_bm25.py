from copy import deepcopy
from datetime import datetime, timezone

import pytest

from utas_research_assistant.models import ProjectRecord
from utas_research_assistant.retrieval.bm25 import BM25Retriever
from utas_research_assistant.retrieval.corpus import build_corpus, tokenize
from utas_research_assistant.retrieval.project_documents import project_document


def chunk(text, identifier="one", category="faq"):
    return dict(
        chunk_id=identifier, document_id=f"doc-{identifier}", title=f"Guide {identifier}",
        category=category, text=text, source_type="html", source_url="https://www.utas.edu.au/",
        local_filename=f"{identifier}.html", page_number=None, chunk_index=0,
        word_count=len(text.split()),
    )


def project(identifier="1"):
    return project_document(ProjectRecord(
        project_id=identifier, title="Phishing detection", description="Detect phishing attacks.",
        source_url=f"https://www.utas.edu.au/research/degrees/available-projects?id={identifier}",
        fetched_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
        primary_supervisor="Doctor Example", research_categories=["ICT"],
    )).model_dump(mode="json")


def test_unicode_tokenization():
    assert tokenize("PhD: Café, naïve! 学生 AI/ML self_funded 2026.") == [
        "phd", "café", "naïve", "学生", "ai", "ml", "self", "funded", "2026",
    ]


def test_corpus_deduplication_and_full_provenance():
    rows = [chunk("English requirements"), chunk("English requirements", "two", "entry_requirements")]
    before = deepcopy(rows)
    corpus = build_corpus(rows, [project()])
    assert corpus.summary == {"raw_item_count": 3, "indexed_item_count": 2, "duplicates_removed": 1}
    general, research = corpus.items
    assert general.item_type == "general_chunk"
    assert research.item_type == "research_project"
    assert general.text == "English requirements"
    assert general.tokens == ["english", "requirements"]
    for source, provenance in zip(rows, general.provenance):
        assert {k: v for k, v in provenance.items() if k != "item_type"} == {k: v for k, v in source.items() if k != "text"}
    assert rows == before


def test_only_exact_general_text_is_deduplicated():
    corpus = build_corpus([
        chunk("Some text"), chunk("Some  text", "two"), chunk("some text", "three"),
    ], [project("1"), project("2")])
    assert len(corpus.items) == 5
    # Identical project text is not enough to merge different projects.
    assert corpus.items[-1].text == corpus.items[-2].text


def test_search_ranks_and_preserves_metadata():
    corpus = build_corpus([
        chunk("English language requirements", "english"), chunk("Tuition fees", "fees"),
        chunk("Marine ecology", "marine"), chunk("Visa applications", "visa"),
    ], [project()])
    retriever = BM25Retriever(corpus)
    result, = retriever.search("phishing")
    assert result["rank"] == 1 and result["score"] > 0
    assert result["item_type"] == "research_project"
    assert result["project_id"] == "1"
    assert result["primary_supervisor"] == "Doctor Example"
    assert result["research_categories"] == ["ICT"]
    assert result["source_url"].endswith("?id=1")
    assert result["provenance"][0]["document_id"] == project()["document_id"]
    general, = retriever.search("English")
    assert general["category"] == "faq"
    assert general["text"] == "English language requirements"
    assert general["project_id"] is None
    results = retriever.search("phishing English fees")
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
    assert [r["score"] for r in results] == sorted([r["score"] for r in results], reverse=True)
    assert results == retriever.search("phishing English fees")


def test_search_retains_all_duplicate_sources():
    retriever = BM25Retriever(build_corpus([
        chunk("English"), chunk("English", "two", "entry_requirements"),
        chunk("Physics", "three"), chunk("Chemistry", "four"),
    ], []))
    results = retriever.search("English")
    assert len(results) == 1
    assert len(results[0]["provenance"]) == 2
    assert {p["category"] for p in results[0]["provenance"]} == {"faq", "entry_requirements"}


@pytest.mark.parametrize("query", ["", "  ", "?!", "nonexistentword"])
def test_empty_and_unmatched_queries(query):
    assert BM25Retriever(build_corpus([chunk("English")], [])).search(query) == []


def test_top_k_and_empty_corpus():
    retriever = BM25Retriever(build_corpus([chunk("shared English"), chunk("shared physics", "two")], []))
    assert len(retriever.search("shared", top_k=1)) == 1
    assert len(retriever.search("shared", top_k=100)) == 2
    assert retriever.search("shared", top_k=0) == []
    assert retriever.search("shared", top_k=-1) == []
    with pytest.raises(ValueError, match="integer"):
        retriever.search("shared", top_k=1.5)
    assert BM25Retriever(build_corpus([], [])).search("English") == []
