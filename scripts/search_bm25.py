"""Search locally: python scripts/search_bm25.py 'user question' --top-k 5."""

import argparse
from pathlib import Path
import sys

from utas_research_assistant.retrieval.bm25 import BM25Retriever
from utas_research_assistant.retrieval.corpus import load_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    try:
        corpus = load_corpus(Path(__file__).resolve().parents[1] / "data/processed")
        results = BM25Retriever(corpus).search(args.query, args.top_k)
    except (OSError, ValueError) as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1
    if not results:
        print("No matching results.")
    for result in results:
        project = f" | project {result['project_id']}" if result["project_id"] else ""
        print(f"{result['rank']}. {result['score']:.3f} | {result['item_type']}{project} | {result['title']}")
        print(f"   {result['source_url'] or '(local source)'} | {len(result['provenance'])} provenance record(s)")
        print(f"   {' '.join(result['text'].split())[:240]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
