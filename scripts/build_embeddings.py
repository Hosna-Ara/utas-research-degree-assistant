"""Build CPU embeddings; first use downloads the selected open-source model."""

import argparse
import json
from pathlib import Path
import sys

from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.semantic import build_embeddings, configured_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=configured_model())
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    processed = Path(__file__).resolve().parents[1] / "data/processed"
    try:
        metadata = build_embeddings(load_corpus(processed), processed,
                                    model_name=args.model, batch_size=args.batch_size)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Embedding build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in metadata.items() if k not in {"items", "embedding_items"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
