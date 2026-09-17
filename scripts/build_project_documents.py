"""Build local project retrieval documents: python scripts/build_project_documents.py."""

from collections import Counter
import json
from pathlib import Path
import sys

from utas_research_assistant.models import ProjectRecord
from utas_research_assistant.retrieval.project_documents import build_project_documents

PROCESSED = Path(__file__).resolve().parents[1] / "data/processed"


def main() -> int:
    try:
        rows = json.loads((PROCESSED / "projects.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("projects.json must contain an array")
        projects = []
        for index, row in enumerate(rows, 1):
            try:
                projects.append(ProjectRecord.model_validate(row))
            except ValueError as exc:
                raise ValueError(f"Invalid project at row {index}: {exc}") from exc
        documents = build_project_documents(projects)
        (PROCESSED / "project_documents.json").write_text(
            json.dumps([d.model_dump(mode="json") for d in documents], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"Project document build failed: {exc}", file=sys.stderr)
        return 1
    counts = Counter(d.funding_status for d in documents)
    print(f"Project retrieval documents: {len(documents)}")
    for status in ("funded", "no_stipend", "unknown"):
        print(f"{status}: {counts[status]}")
    print("Sample project documents:")
    for document in documents[:3]:
        print(json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
