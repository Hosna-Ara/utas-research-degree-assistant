"""Build chunks from local general documents only: python scripts/build_chunks.py."""

import json
from pathlib import Path
import sys

from utas_research_assistant.ingestion.parse_general_docs import GeneralDocumentRecord
from utas_research_assistant.retrieval.chunking import build_chunks

PROCESSED = Path(__file__).resolve().parents[1] / "data/processed"


def main() -> int:
    try:
        rows = json.loads((PROCESSED / "general_documents.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("general_documents.json must contain an array")
        documents = []
        for index, row in enumerate(rows, 1):
            try:
                documents.append(GeneralDocumentRecord.model_validate(row))
            except ValueError as exc:
                raise ValueError(f"Invalid general document at row {index}: {exc}") from exc
        chunks, manifest = build_chunks(documents)
        (PROCESSED / "general_chunks.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in chunks], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (PROCESSED / "chunk_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        print(f"Chunk build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nSample chunks:")
    for chunk in chunks[:2]:
        print(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
