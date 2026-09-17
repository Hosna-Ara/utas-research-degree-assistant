"""Inspect BM25 and semantic top-five results without judging relevance."""

import json
from pathlib import Path
import sys


def result_identity(result: dict) -> tuple:
    # Same titles can refer to different chunks: compare source identities, not titles.
    provenance = result["provenance"][0]
    return result["item_type"], provenance.get("chunk_id", provenance.get("document_id"))


def main() -> int:
    processed = Path(__file__).resolve().parents[1] / "data/processed"
    try:
        bm25 = json.loads((processed / "bm25_probe_results.json").read_text())
        semantic = json.loads((processed / "semantic_probe_results.json").read_text())
        semantic_by_query = {p["query"]: p for p in semantic["probes"]}
        if {p["query"] for p in bm25["probes"]} != set(semantic_by_query):
            raise ValueError("Probe query sets differ")
        for probe in bm25["probes"]:
            other = semantic_by_query[probe["query"]]
            print(f"\n{probe['label']}. {probe['query']}")
            for method, results in (("BM25", probe["results"]), ("Semantic", other["results"])):
                print(f"{method}:")
                for result in results[:5]:
                    source = result["provenance"][0]
                    detail = f"project {result['project_id']}" if result.get("project_id") else f"chunk {source['chunk_index']}"
                    print(f"  {result['rank']}. {result['title']} ({detail})")
            overlap = {result_identity(r) for r in probe["results"][:5]} & {result_identity(r) for r in other["results"][:5]}
            print(f"Top-5 overlap: {len(overlap)}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
