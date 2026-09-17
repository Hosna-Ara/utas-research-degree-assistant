"""Run the exact BM25 probe questions against cached local semantic vectors."""

import json
from pathlib import Path
import sys

from probe_bm25 import QUESTIONS
from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.semantic import SemanticRetriever


def main() -> int:
    processed = Path(__file__).resolve().parents[1] / "data/processed"
    try:
        corpus = load_corpus(processed)
        retriever = SemanticRetriever(corpus, processed)
        probes = [{"label": chr(65 + i), "query": query, "results": retriever.search(query)}
                  for i, query in enumerate(QUESTIONS)]
        report = {"corpus": corpus.summary, "model_name": retriever.metadata["model_name"], "probes": probes}
        (processed / "semantic_probe_results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Semantic probe failed: {exc}", file=sys.stderr)
        return 1
    for probe in probes:
        print(f"\n{probe['label']}. {probe['query']}")
        for result in probe["results"]:
            project = f" | project {result['project_id']}" if result["project_id"] else ""
            print(f"{result['rank']}. {result['score']:.4f} | {result['item_type']} | {result['title']}{project}")
    print("\nSaved: data/processed/semantic_probe_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
