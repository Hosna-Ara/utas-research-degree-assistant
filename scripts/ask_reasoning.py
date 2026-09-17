"""Plan and route a question locally without generating a natural-language answer."""

import argparse
import json
from pathlib import Path
import sys

from rdflib import Graph

from utas_research_assistant.query.planner import ReasoningPlanner
from utas_research_assistant.query.reasoning_router import ReasoningRouter
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
        graph = Graph().parse(processed / "utas_research_graph.ttl", format="turtle")
        router = ReasoningRouter(ReasoningPlanner(), HybridRetriever(
            corpus, None, SemanticRetriever(corpus, processed)), graph, top_k=args.top_k)
        evidence = router.route(args.question)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Reasoning route failed: {exc}", file=sys.stderr)
        return 1
    print(f"Method: {evidence['reasoning_method']} | planner: {evidence['planner_method']} | "
          f"intent: {evidence['interpreted_intent']}")
    print(f"Graph operation: {evidence['graph_operation_used'] or '(none)'}")
    print(f"Constraints: {json.dumps(evidence['extracted_constraints'], ensure_ascii=False) or '(none)'}")
    print(f"Candidates: {evidence['candidate_count_before']} -> {evidence['candidate_count_after']}")
    for result in evidence["ranked_retrieval_evidence"]:
        identifier = f" | project {result['project_id']}" if result.get("project_id") else ""
        print(f"  Retrieval: {result['title']}{identifier} | {result.get('source_url')}")
    graph_result = evidence["graph_result"]
    if isinstance(graph_result, list):
        for result in graph_result[:5]:
            if isinstance(result, dict):
                print(f"  Graph: {result.get('title') or result.get('supervisor') or result.get('category') or result}")
            else:
                print(f"  Graph: {result}")
        if len(graph_result) > 5:
            print(f"  Graph: … {len(graph_result)} result rows total")
    elif graph_result is not None:
        print(f"  Graph: {json.dumps(graph_result, ensure_ascii=False, default=str)[:700]}")
    print(f"Tool trace: {json.dumps(evidence['tool_trace'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
