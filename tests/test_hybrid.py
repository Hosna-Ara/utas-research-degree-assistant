from datetime import datetime, timezone

import pytest

from utas_research_assistant.retrieval.bm25 import BM25Retriever
from utas_research_assistant.retrieval.corpus import Corpus, CorpusItem
from utas_research_assistant.retrieval.hybrid import HybridRetriever, rrf_score
from utas_research_assistant.retrieval.project_documents import ProjectDocument


def project(i, title, text, **changes):
    values = dict(
        document_id=f"project-{i}", project_id=str(i), title=title, text=text,
        source_url=f"https://www.utas.edu.au/research/degrees/available-projects?id={i}",
        status="Applications open", degree_types=["PhD"], student_types=["International"],
        location="Hobart", scholarship_text="$34,315 pa", funding_status="funded",
        closing_date="1 October 2026", primary_supervisor="Doctor Example",
        research_categories=["Information and Communication Technology"], description=text,
    )
    return ProjectDocument(**(values | changes))


def make_corpus():
    items = []
    for item_type, i, title, text in [
        ("general_chunk", "g1", "Electric cars", "electric cars and transport"),
        ("general_chunk", "g2", "Forest habitats", "forest habitat conservation"),
        ("research_project", "11", "Phishing detection", "machine learning phishing detection"),
        ("research_project", "12", "Marine biology", "marine biology and transport"),
    ]:
        metadata = (dict(project( int(i), title, text).model_dump(mode="json"))
                    if item_type == "research_project" else {
                        "chunk_id": i, "document_id": f"doc-{i}", "title": title,
                        "category": "faq", "source_url": "https://www.utas.edu.au/",
                        "local_filename": "faq.html", "page_number": None,
                    })
        items.append(CorpusItem(item_type, text, text.split(), metadata, [metadata.copy()]))
    return Corpus(items, 4)


class OrderedSemantic:
    def __init__(self, corpus, order=None, omit=()):
        self.corpus = corpus
        self.order = order or [2, 0, 3, 1]
        self.omit = set(omit)

    def search(self, query, top_k=5, *, candidate_indices=None):
        candidates = set(range(len(self.corpus.items))) if candidate_indices is None else candidate_indices
        indices = [i for i in self.order if i in candidates and i not in self.omit]
        result = []
        for rank, index in enumerate(indices[:top_k], 1):
            item = self.corpus.items[index]
            m = item.metadata
            result.append({
                "rank": rank, "score": 1 - rank / 10, "item_type": item.item_type,
                "title": m["title"], "text": item.text, "source_url": m.get("source_url"),
                "project_id": m.get("project_id"), "primary_supervisor": m.get("primary_supervisor"),
                "research_categories": m.get("research_categories"), "category": m.get("category"),
                "provenance": item.provenance,
            })
        return result


@pytest.fixture
def retriever():
    corpus = make_corpus()
    return HybridRetriever(corpus, BM25Retriever(corpus), OrderedSemantic(corpus), rrf_k=10)


def test_rrf_calculation_and_single_component():
    assert rrf_score(bm25_rank=1, semantic_rank=2, k=60) == pytest.approx(1 / 61 + 1 / 62)
    assert rrf_score(bm25_rank=1, semantic_rank=None, k=10) == pytest.approx(1 / 11)
    assert rrf_score(bm25_rank=None, semantic_rank=2, k=10) == pytest.approx(1 / 12)
    with pytest.raises(ValueError):
        rrf_score(bm25_rank=1, semantic_rank=2, k=0)


def test_shared_and_single_source_result_fusion(retriever):
    results = retriever.search("transport", top_k=5)
    by_title = {r["title"]: r for r in results}
    assert by_title["Marine biology"]["bm25_rank"] is not None
    assert by_title["Marine biology"]["semantic_rank"] == 3
    assert by_title["Marine biology"]["rrf_score"] == pytest.approx(
        1 / (10 + by_title["Marine biology"]["bm25_rank"]) + 1 / 13)
    assert by_title["Electric cars"]["semantic_rank"] == 2
    assert by_title["Electric cars"]["bm25_rank"] is not None
    assert by_title["Forest habitats"]["bm25_rank"] is None
    # A component-only result has a finite score and a missing rank for the other component.
    assert by_title["Forest habitats"]["semantic_rank"] == 4
    semantic_without_marine = HybridRetriever(
        retriever.corpus, retriever.bm25,
        OrderedSemantic(retriever.corpus, omit={3}), rrf_k=10,
    )
    only_bm25, = [r for r in semantic_without_marine.search("transport") if r["title"] == "Marine biology"]
    assert only_bm25["bm25_rank"] is not None and only_bm25["semantic_rank"] is None


def test_component_absence_uses_rank_only_and_stable_order(retriever):
    one_source = HybridRetriever(retriever.corpus, retriever.bm25,
                                 OrderedSemantic(retriever.corpus, omit=range(4)), rrf_k=10)
    results = one_source.search("phishing", top_k=3)
    assert all(r["semantic_rank"] is None for r in results)
    assert results[0]["bm25_rank"] == 1
    assert one_source.search("phishing") == one_source.search("phishing")


def test_scope_projects_and_general(retriever):
    assert all(r["item_type"] == "research_project" for r in retriever.search("transport", scope="projects"))
    assert all(r["item_type"] == "general_chunk" for r in retriever.search("transport", scope="general"))
    with pytest.raises(ValueError, match="scope"):
        retriever.search("transport", scope="other")
    with pytest.raises(ValueError, match="general-document"):
        retriever.search("transport", scope="general", filters={"location": "Hobart"})


def test_filters_are_applied_before_both_rankers(retriever):
    candidates = retriever.candidate_indices("projects", {"project_id": "11"})
    assert candidates == [2]
    assert retriever.search("marine", scope="projects", filters={"project_id": "11"})[0]["project_id"] == "11"
    assert retriever.search("marine", scope="projects", filters={"project_id": "missing"}) == []
    assert retriever.search("transport", scope="all", filters={"project_id": "11"})
    by_graph_candidates = retriever.search("transport", scope="projects", candidate_project_ids={"12"})
    assert by_graph_candidates
    assert {result["project_id"] for result in by_graph_candidates} == {"12"}
    assert retriever.search("transport", scope="projects", candidate_project_ids=set()) == []
    with pytest.raises(ValueError, match="Unknown filters"):
        retriever.search("x", scope="projects", filters={"bad": "value"})


def test_metadata_preserved(retriever):
    results = retriever.search("phishing", scope="projects")
    result, = [r for r in results if r["project_id"] == "11"]
    assert result["degree_types"] == ["PhD"]
    assert result["student_types"] == ["International"]
    assert result["location"] == "Hobart"
    assert result["funding_status"] == "funded"
    assert result["status"] == "Applications open"
    assert result["primary_supervisor"] == "Doctor Example"
    assert result["research_categories"] == ["Information and Communication Technology"]
    assert result["source_url"].endswith("?id=11")
    assert result["provenance"] == retriever.corpus.items[2].provenance


def test_top_k_and_empty_candidates(retriever):
    assert len(retriever.search("transport", top_k=1)) == 1
    assert retriever.search("transport", top_k=0) == []
    assert retriever.search("transport", top_k=-1) == []
    with pytest.raises(ValueError, match="integer"):
        retriever.search("transport", top_k=1.5)
