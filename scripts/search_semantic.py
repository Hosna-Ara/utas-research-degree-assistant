"""Search cached local embeddings: python scripts/search_semantic.py 'question'."""

import argparse
from pathlib import Path
import sys

from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.semantic import SemanticRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    processed = Path(__file__).resolve().parents[1] / "data/processed"
    try:
        results = SemanticRetriever(load_corpus(processed), processed).search(args.query, args.top_k)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Semantic search failed: {exc}", file=sys.stderr)
        return 1
    if not results:
        print("No results.")
    for result in results:
        project = f" | project {result['project_id']}" if result['project_id'] else ""
        print(f"{result['rank']}. {result['score']:.4f} | {result['item_type']} | {result['title']}{project}")
        print(f"   {result['source_url'] or '(local source)'}")
        print(f"   {' '.join(result['text'].split())[:240]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
