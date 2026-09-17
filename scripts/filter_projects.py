"""Filter local project metadata, or run the four fixed cases with --probe."""

import argparse
import json
from pathlib import Path
import sys

from utas_research_assistant.retrieval.filters import filter_projects
from utas_research_assistant.retrieval.project_documents import ProjectDocument

PROBES = {
    "A": dict(student_type="International", research_category="ICT", degree_type="PhD", funding_status="funded", status="open"),
    "B": dict(degree_type="Master by Research", location="Hobart", status="open"),
    "C": dict(student_type="International", research_category="AI/ICT", funding_status="funded"),
    "D": dict(funding_status="no_stipend"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    for option, destination in (("degree", "degree_type"), ("student-type", "student_type"),
                                ("location", "location"), ("funding", "funding_status"),
                                ("status", "status"), ("research-category", "research_category"),
                                ("supervisor", "supervisor"), ("project-id", "project_id")):
        parser.add_argument(f"--{option}", dest=destination)
    args = vars(parser.parse_args())
    probe = args.pop("probe")
    try:
        source = Path(__file__).resolve().parents[1] / "data/processed/project_documents.json"
        projects = [ProjectDocument.model_validate(row) for row in json.loads(source.read_text())]
        for name, constraints in (PROBES if probe else {"Filters": args}).items():
            results = filter_projects(projects, **constraints)
            print(f"\n{name}: {json.dumps(constraints)}\nMatches: {len(results)}")
            for project in results[:10]:
                print(json.dumps(project.model_dump(mode="json", include={
                    "project_id", "title", "degree_types", "student_types", "location", "funding_status",
                    "scholarship_text", "status", "research_categories", "primary_supervisor",
                }), ensure_ascii=False))
    except (OSError, ValueError) as exc:
        print(f"Filtering failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
