"""Parse the local snapshot. Run: python scripts/parse_projects_snapshot.py.

fetched_at uses the saved file's modification time as an acquisition-time proxy;
parsed_at is the actual parser run time. No network access is performed.
CSV list fields are JSON arrays to preserve their values unambiguously.
"""

import csv
from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utas_research_assistant.ingestion.parse_projects import parse_projects
from utas_research_assistant.models import ProjectRecord

SNAPSHOT = ROOT / "data/raw/utas_available_projects_2026-09-14.html"
OUTPUT = ROOT / "data/processed"
SAMPLE_FIELDS = (
    "project_id", "title", "degree_types", "student_types",
    "scholarship_text", "primary_supervisor",
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if not SNAPSHOT.is_file():
            raise FileNotFoundError(f"Snapshot file does not exist: {SNAPSHOT}")
        snapshot_date = date.fromisoformat(SNAPSHOT.stem.rsplit("_", 1)[1])
        parsed_at = datetime.now(timezone.utc)
        fetched_at = datetime.fromtimestamp(SNAPSHOT.stat().st_mtime, timezone.utc)
        records = parse_projects(SNAPSHOT.read_text(encoding="utf-8"), fetched_at=fetched_at)
        rows = [record.model_dump(mode="json") for record in records]
        manifest = {
            "source_file": SNAPSHOT.relative_to(ROOT).as_posix(),
            "snapshot_date": snapshot_date.isoformat(),
            "parsed_at": parsed_at.isoformat(),
            "total_projects": len(records),
            "projects_with_id": sum(bool(r.project_id) for r in records),
            "projects_with_supervisor": sum(bool(r.primary_supervisor) for r in records),
            "projects_with_scholarship": sum(bool(r.scholarship_text) for r in records),
            "projects_accepting_international_students": sum(
                any(value.casefold() == "international" for value in r.student_types)
                for r in records
            ),
            "projects_with_closing_date": sum(bool(r.closing_date) for r in records),
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "projects.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with (OUTPUT / "projects.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ProjectRecord.model_fields))
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                    for key, value in row.items()
                })
        (OUTPUT / "snapshot_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, UnicodeError, ValueError) as exc:
        logging.error("Snapshot parsing failed: %s", exc)
        return 1

    print(f"Snapshot: {manifest['source_file']}")
    print(f"Total projects parsed: {manifest['total_projects']}")
    print(f"Unique project IDs: {len({r.project_id for r in records if r.project_id})}")
    print(f"With supervisors: {manifest['projects_with_supervisor']}")
    print(f"With scholarship information: {manifest['projects_with_scholarship']}")
    print(f"Accepting international students: {manifest['projects_accepting_international_students']}")
    print(f"With closing dates: {manifest['projects_with_closing_date']}")
    print("Sample records:")
    for row in rows[:3]:
        print(json.dumps({key: row[key] for key in SAMPLE_FIELDS}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
