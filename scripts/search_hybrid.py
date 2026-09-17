"""Search with rank fusion: python scripts/search_hybrid.py 'question' --scope projects."""

import argparse
from pathlib import Path
import sys

from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.hybrid import HybridRetriever
from utas_research_assistant.retrieval.semantic import SemanticRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--scope", choices=("all", "projects", "general"), default="all")
    parser.add_argument("--top-k", type=int, default=5)
    for option, dest in (("degree", "degree_type"), ("student-type", "student_type"),
                         ("location", "location"), ("funding", "funding_status"),
                         ("status", "status"), ("research-category", "research_category"),
                         ("supervisor", "supervisor"), ("project-id", "project_id")):
        parser.add_argument(f"--{option}", dest=dest)
    args = vars(parser.parse_args())
    query, scope, top_k = args.pop("query"), args.pop("scope"), args.pop("top_k")
    filters = {key: value for key, value in args.items() if value is not None}
    processed = Path(__file__).resolve().parents[1] / "data/processed"
    try:
        corpus = load_corpus(processed)
        retriever = HybridRetriever(corpus, None, SemanticRetriever(corpus, processed))
        candidates = retriever.candidate_indices(scope, filters or None)
        results = retriever.search(query, top_k, scope, filters or None)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Hybrid search failed: {exc}", file=sys.stderr)
        return 1
    print(f"Candidates: {len(candidates)} | scope: {scope} | filters: {filters or '(none)'}")
    if not results:
        print("No results.")
    for item in results:
        print(f"{item['rank']}. RRF {item['rrf_score']:.5f} | BM25 {item['bm25_rank']} "
              f"({item['bm25_score'] if item['bm25_score'] is not None else '—'}) | Semantic "
              f"{item['semantic_rank']} ({item['semantic_score'] if item['semantic_score'] is not None else '—'})")
        project = f" | project {item['project_id']}" if item["project_id"] else ""
        print(f"   {item['item_type']} | {item['title']}{project}")
        print(f"   {item['source_url'] or '(local source)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
