"""Run fixed local BM25 probes: python scripts/probe_bm25.py."""

import json
from pathlib import Path
import sys

from utas_research_assistant.retrieval.bm25 import BM25Retriever
from utas_research_assistant.retrieval.corpus import load_corpus

QUESTIONS = [
    "What are the English language requirements for a PhD?",
    "How do I apply for a research degree?",
    "What documents do I need to submit with my application?",
    "What scholarships are available for research students?",
    "machine learning artificial intelligence PhD projects",
    "phishing detection research project",
    "international student ICT PhD scholarship",
    "Master by Research projects in Hobart",
]


def main() -> int:
    processed = Path(__file__).resolve().parents[1] / "data/processed"
    try:
        corpus = load_corpus(processed)
        retriever = BM25Retriever(corpus)
        probes = [
            {"label": chr(65 + index), "query": query, "results": retriever.search(query)}
            for index, query in enumerate(QUESTIONS)
        ]
        report = {"corpus": corpus.summary, "probes": probes}
        (processed / "bm25_probe_results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        print(f"BM25 probe failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(corpus.summary, indent=2))
    for probe in probes:
        print(f"\n{probe['label']}. {probe['query']}")
        for result in probe["results"]:
            project = f" | project {result['project_id']}" if result["project_id"] else ""
            print(f"{result['rank']}. {result['score']:.3f} | {result['item_type']} | {result['title']}{project}")
    print("\nSaved: data/processed/bm25_probe_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
