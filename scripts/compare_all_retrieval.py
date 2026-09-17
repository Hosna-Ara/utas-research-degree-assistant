"""Inspect top-five BM25, semantic, and hybrid results side by side."""

import json
from pathlib import Path
import sys


def identity(result):
    source = result["provenance"][0]
    return result["item_type"], source.get("chunk_id", result.get("project_id"))


def main() -> int:
    processed = Path(__file__).resolve().parents[1] / "data/processed"
    try:
        reports = {name: json.loads((processed / f"{name}_probe_results.json").read_text())
                   for name in ("bm25", "semantic", "hybrid")}
        lookups = {name: {probe["query"]: probe for probe in report["probes"]}
                   for name, report in reports.items()}
        queries = list(lookups["hybrid"])
        if any(set(lookup) != set(queries) for lookup in lookups.values()):
            raise ValueError("Probe query sets differ")
        for label, query in enumerate(queries):
            hybrid_probe = lookups["hybrid"][query]
            print(f"\n{chr(65 + label)}. {query}")
            if label >= 6:
                print("Hybrid uses structured project constraints; this is not a ranking-only comparison.")
            sets = []
            for method in ("bm25", "semantic", "hybrid"):
                probe = lookups[method][query]
                print(f"{method.upper()} (scope={probe.get('scope', 'all')}):")
                for result in probe["results"][:5]:
                    extra = f"project {result['project_id']}" if result.get("project_id") else "general"
                    ranks = (f" | BM25 {result['bm25_rank']}, semantic {result['semantic_rank']}"
                             if method == "hybrid" else "")
                    print(f"  {result['rank']}. {result['title']} ({extra}){ranks}")
                sets.append({identity(result) for result in probe["results"][:5]})
            print(f"Top-five overlap BM25/semantic/hybrid: {len(sets[0] & sets[1])}/"
                  f"{len(sets[0] & sets[2])}/{len(sets[1] & sets[2])}")
            if label in (6, 7):
                print(f"Project candidates: {hybrid_probe['candidate_count_before_filters']} -> "
                      f"{hybrid_probe['candidate_count_after_filters']}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"Retrieval comparison failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
