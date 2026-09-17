"""Ingest user-authored local documents without making web requests."""

import json
from pathlib import Path
import sys

from utas_research_assistant.ingestion.parse_local_docs import parse_directory

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/private"
OUTPUT = ROOT / "data/processed"


def main() -> int:
    try:
        documents, chunks, manifest = parse_directory(SOURCE)
    except (OSError, ValueError) as exc:
        print(f"Local document ingestion failed: {exc}", file=sys.stderr)
        return 1
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "local_documents.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in documents], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "local_chunks.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in chunks], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "local_documents_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not documents:
        print(f"No local documents found. Add HTML, HTM, TXT, PDF, or DOCX files to {SOURCE}.")
    return 1 if manifest["empty_or_failed_files"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
