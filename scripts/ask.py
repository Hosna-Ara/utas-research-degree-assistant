"""Answer a question using routed local evidence and Ollama."""

import argparse
from pathlib import Path
import sys

from rdflib import Graph

from utas_research_assistant.generation.answer_generator import AnswerGenerator
from utas_research_assistant.query.planner import ReasoningPlanner
from utas_research_assistant.query.reasoning_router import ReasoningRouter
from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.hybrid import HybridRetriever
from utas_research_assistant.retrieval.semantic import SemanticRetriever


def make_router(processed: Path, top_k: int = 5) -> ReasoningRouter:
    corpus = load_corpus(processed)
    graph = Graph().parse(processed / "utas_research_graph.ttl", format="turtle")
    retriever = HybridRetriever(corpus, None, SemanticRetriever(corpus, processed))
    return ReasoningRouter(ReasoningPlanner(), retriever, graph, top_k=top_k)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    processed = Path(__file__).resolve().parents[1] / "data/processed"
    try:
        evidence = make_router(processed, args.top_k).route(args.question)
        generator = AnswerGenerator()
        response = generator.generate(args.question, evidence)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Ask failed: {exc}", file=sys.stderr)
        return 1
    print(f"QUESTION\n{response.question}\n")
    print(f"ANSWER\n{response.answer}\n")
    print("SOURCES")
    for source in response.sources:
        detail = f" — {source['title']}"
        if source.get("project_id"):
            detail = f" — Project {source['project_id']} — {source['title']}"
        print(f"[{source['citation_id']}]{detail}")
        if source.get("source_url"):
            print(f"     {source['source_url']}")
    if response.tool_used:
        print(f"\nTOOL USED\n{response.tool_used}\n{response.tool_result}")
    print(f"\nREASONING METHOD\n{response.reasoning_method} ({response.planner_method})")
    if generator.last_error:
        print(f"\nGeneration fallback: {generator.last_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
