"""Probe query planning and evidence retrieval without generating answers."""

import json
from pathlib import Path
import sys

from utas_research_assistant.query.planner import QueryPlanner
from utas_research_assistant.query.router import RetrievalRouter
from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.hybrid import HybridRetriever
from utas_research_assistant.retrieval.semantic import SemanticRetriever

QUESTIONS = [
    "What English score do I need for a PhD?",
    "How do I apply for a research degree?",
    "What documents are required for my application?",
    "What scholarships are available?",
    "Find AI and machine learning PhD projects.",
    "Is there a project about phishing detection?",
    "I am an international student looking for a funded ICT PhD.",
    "Show me Master by Research projects in Hobart.",
    "I want funded AI research opportunities in Hobart for international students.",
    "Who supervises project 12259?",
]


def main() -> int:
    processed = Path(__file__).resolve().parents[1] / "data/processed"
    try:
        corpus = load_corpus(processed)
        planner = QueryPlanner()
        available = planner.llm_available
        router = RetrievalRouter(planner, HybridRetriever(corpus, None, SemanticRetriever(corpus, processed)))
        rows = []
        for label, question in zip("ABCDEFGHIJ", QUESTIONS):
            outcome = router.route(question)
            plan = outcome["plan"].model_dump(mode="json")
            rows.append({"label": label, **outcome, "plan": plan})
        report = {
            "ollama_available": available,
            "ollama_model": getattr(planner.provider, "model", None),
            "corpus": corpus.summary,
            "probes": rows,
        }
        (processed / "query_planner_probe_results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Query planner probe failed: {exc}", file=sys.stderr)
        return 1
    print(f"Local Ollama model available: {available} ({report['ollama_model']})")
    for row in rows:
        plan = row["plan"]
        print(f"\n{row['label']}. {row['question']}\n  Plan: {json.dumps(plan, ensure_ascii=False)}")
        print(f"  Candidates: {row['candidate_count_before_filters']} -> {row['candidate_count_after_filters']}")
        for result in row["results"]:
            project = f" | project {result['project_id']}" if result["project_id"] else ""
            print(f"  {result['rank']}. {result['title']}{project} (RRF {result['rrf_score']:.5f})")
    print("\nSaved: data/processed/query_planner_probe_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
