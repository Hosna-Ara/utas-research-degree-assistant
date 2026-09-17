"""Plan a question and retrieve evidence locally: python scripts/ask_retrieval.py 'question'."""

import argparse
import json
from pathlib import Path
import sys

from utas_research_assistant.query.planner import QueryPlanner
from utas_research_assistant.query.router import RetrievalRouter
from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.hybrid import HybridRetriever
from utas_research_assistant.retrieval.semantic import SemanticRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    processed = Path(__file__).resolve().parents[1] / "data/processed"
    try:
        corpus = load_corpus(processed)
        semantic = SemanticRetriever(corpus, processed)
        planner = QueryPlanner()
        router = RetrievalRouter(planner, HybridRetriever(corpus, None, semantic))
        outcome = router.route(args.question, top_k=args.top_k)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        return 1
    plan = outcome["plan"]
    print(f"Intent: {plan.intent} | scope: {plan.scope} | planner: {plan.planner_method}")
    if outcome["planner_error"]:
        print(f"Planner note: {outcome['planner_error']}")
    print(f"Search query: {plan.search_query}")
    print(f"Filters: {json.dumps(plan.project_filters(), ensure_ascii=False) or '(none)'}")
    print(f"Candidates: {outcome['candidate_count_before_filters']} -> {outcome['candidate_count_after_filters']}")
    for result in outcome["results"]:
        identifier = f" | project {result['project_id']}" if result["project_id"] else ""
        print(f"\n{result['rank']}. {result['title']}{identifier} | RRF {result['rrf_score']:.5f}")
        print(f"   {result['source_url'] or '(local source)'}")
        print(f"   {' '.join(result['text'].split())[:420]}")
    if not outcome["results"]:
        print("No evidence matched the plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
