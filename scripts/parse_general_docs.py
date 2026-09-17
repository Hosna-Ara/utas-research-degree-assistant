"""Parse local files only: python scripts/parse_general_docs.py."""

import json
from pathlib import Path
import sys

from utas_research_assistant.ingestion.parse_general_docs import parse_directory

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    source = ROOT / "data/raw/general"
    output = ROOT / "data/processed"
    try:
        records, manifest = parse_directory(source)
        output.mkdir(parents=True, exist_ok=True)
        (output / "general_documents.json").write_text(
            json.dumps([r.model_dump(mode="json") for r in records], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "general_documents_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        print(f"General document ingestion failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not records:
        print("No readable document records extracted.", file=sys.stderr)
    return 1 if manifest["failed_files"] or not records else 0


if __name__ == "__main__":
    raise SystemExit(main())
