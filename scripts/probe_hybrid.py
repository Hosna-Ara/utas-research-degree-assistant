"""Run the eight fixed hybrid probes and save complete component ranks."""

import json
from pathlib import Path
import sys

from probe_bm25 import QUESTIONS
from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.hybrid import HybridRetriever
from utas_research_assistant.retrieval.semantic import SemanticRetriever

SCOPES = ["general"] * 4 + ["projects"] * 4
FILTERS = [None] * 6 + [
    dict(degree_type="PhD", student_type="International", research_category="Information and Communication Technology",
         funding_status="funded", status="Applications open"),
    dict(degree_type="Master by Research", location="Hobart", status="Applications open"),
]


def main() -> int:
    processed = Path(__file__).resolve().parents[1] / "data/processed"
    try:
        corpus = load_corpus(processed)
        retriever = HybridRetriever(corpus, None, SemanticRetriever(corpus, processed))
        probes = []
        for index, (query, scope, filters) in enumerate(zip(QUESTIONS, SCOPES, FILTERS)):
            candidates = retriever.candidate_indices(scope, filters)
            results = retriever.search(query, 5, scope, filters)
            probes.append({
                "label": chr(65 + index), "query": query, "scope": scope, "filters": filters,
                "candidate_count_before_filters": (
                    sum(item.item_type == "research_project" for item in corpus.items)
                    if scope == "projects" and filters else len(corpus.items)),
                "candidate_count_after_filters": len(candidates), "results": results,
            })
        report = {"corpus": corpus.summary, "rrf_k": retriever.rrf_k, "probes": probes}
        (processed / "hybrid_probe_results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Hybrid probe failed: {exc}", file=sys.stderr)
        return 1
    for probe in probes:
        print(f"\n{probe['label']}. {probe['query']} [{probe['scope']}] filters={probe['filters']}")
        if probe["filters"]:
            print(f"Candidates: {probe['candidate_count_before_filters']} -> {probe['candidate_count_after_filters']}")
        for item in probe["results"]:
            project = f" | project {item['project_id']}" if item["project_id"] else ""
            print(f"{item['rank']}. {item['rrf_score']:.5f} | {item['item_type']} | {item['title']}{project} "
                  f"[BM25 {item['bm25_rank']}, semantic {item['semantic_rank']}]")
    print("\nSaved: data/processed/hybrid_probe_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
