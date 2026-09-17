"""Read-only audit. Run after editable installation: python scripts/audit_projects.py."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys

from pydantic import ValidationError

from utas_research_assistant.models import ProjectRecord

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/projects.json"
REPORT = ROOT / "data/processed/data_quality_report.json"
RANDOM_SEED = 42
SHORT_DESCRIPTION_CHARS = 80
MONEY = re.compile(
    r"(?:[$£€]\s*\d|\b(?:AUD|USD|GBP|EUR)\s*\d|"
    r"\b\d[\d,.]*\s*(?:AUD|USD|GBP|EUR|dollars?)\b)", re.IGNORECASE
)
NON_FUNDING = re.compile(
    r"\b(?:not\s+available|unavailable|no\s+(?:scholarship|stipend|funding)|"
    r"self[\s\-–—]*funded|unfunded|without\s+(?:funding|stipend|scholarship)|"
    r"not\s+funded|not\s+applicable|n\s*/\s*a)\b|^na$|^none$|^nil$",
    re.IGNORECASE,
)
FUNDING_LABELS = {
    "scholarship", "scholarships", "scholarship type", "scholarship amount",
    "scholarship information", "scholarship availability", "funding", "stipend",
    "scholarships and fees", "scholarship details", "click here", "read more",
    "learn more", "view details",
}


def text(value) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value or all(is_empty(item) for item in (
            value.values() if isinstance(value, dict) else value
        ))
    return False


def scholarship_flags(value: str | None) -> dict[str, bool]:
    value = text(value)
    return {
        "monetary": bool(MONEY.search(value)),
        "non_funding": bool(NON_FUNDING.search(value)),
        "label_like": value.casefold().rstrip(":") in FUNDING_LABELS,
    }


def audit(rows: list) -> tuple[dict, list[dict]]:
    """Report invalid rows explicitly; distributions use validated records only."""
    valid = []
    failures = []
    for index, row in enumerate(rows, 1):
        try:
            record = ProjectRecord.model_validate(row)
            valid.append((index, record.model_dump(mode="json")))
        except ValidationError as exc:
            failures.append({
                "row": index,
                "errors": exc.errors(include_url=False, include_context=False),
            })

    # Examine original rows for missing values, even if validation failed.
    missing = {
        field: sum(not isinstance(row, dict) or is_empty(row.get(field)) for row in rows)
        for field in ProjectRecord.model_fields
    }
    ids = Counter(text(row.get("project_id")) for row in rows if isinstance(row, dict))
    ids.pop("", None)
    suspicious = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue  # Already reported as a validation failure.
        project_id = text(row.get("project_id"))
        reasons = []
        if not text(row.get("title")):
            reasons.append("empty title")
        if not project_id:
            reasons.append("missing project ID")
        elif ids[project_id] > 1:
            reasons.append("duplicate project ID")
        if not text(row.get("primary_supervisor")):
            reasons.append("missing supervisor")
        description_length = len(text(row.get("description")))
        if description_length < SHORT_DESCRIPTION_CHARS:
            reasons.append("unusually short description")
        if scholarship_flags(row.get("scholarship_text"))["label_like"]:
            reasons.append("scholarship looks like a label")
        if reasons:
            suspicious.append({
                "row": index, "project_id": project_id,
                "title": row.get("title"), "reasons": reasons,
                "description_characters": description_length,
                "scholarship_text": row.get("scholarship_text"),
            })

    records = [record for _, record in valid]

    def distribution(field: str, *, multiple: bool = False, locations: bool = False) -> dict:
        counts = Counter()
        for record in records:
            value = record[field]
            values = (value or "").split(";") if locations else value if multiple else [value]
            counts.update({text(item) for item in values if text(item)})
        return dict(counts.most_common())

    scholarships = Counter(text(r["scholarship_text"]) for r in records if text(r["scholarship_text"]))
    flags = [scholarship_flags(r["scholarship_text"]) for r in records]
    samples = random.Random(RANDOM_SEED).sample(valid, min(5, len(valid)))
    report = {
        "source_file": SOURCE.relative_to(ROOT).as_posix(),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "distributions": "Validated records only; count each project once per value. Locations split on semicolons.",
            "missing_and_suspicious": "All input rows; missing includes null, blank text, and empty lists.",
            "short_description": f"Fewer than {SHORT_DESCRIPTION_CHARS} characters after whitespace cleaning; a review flag, not an error.",
            "scholarships": "Text heuristics only, not confirmation of funding. Monetary and non-funding counts may overlap. Unique values exclude blanks.",
            "random_seed": RANDOM_SEED,
        },
        "total_projects": len(rows),
        "validated_projects": len(valid),
        "validation_failure_count": len(failures),
        "validation_failures": failures,
        "unique_project_ids": len({r["project_id"].strip() for r in records if r["project_id"].strip()}),
        "unique_supervisors": len({text(r["primary_supervisor"]) for r in records if text(r["primary_supervisor"])}),
        "status_distribution": distribution("status"),
        "degree_type_distribution": distribution("degree_types", multiple=True),
        "student_type_distribution": distribution("student_types", multiple=True),
        "location_distribution": distribution("location", locations=True),
        "research_category_distribution": distribution("research_categories", multiple=True),
        "scholarship_analysis": {
            "unique_values": len(scholarships),
            "top_20_values": dict(scholarships.most_common(20)),
            "records_with_monetary_values": sum(f["monetary"] for f in flags),
            "records_with_non_funding_wording": sum(f["non_funding"] for f in flags),
            "non_funding_values": {v: n for v, n in scholarships.items() if scholarship_flags(v)["non_funding"]},
            "records_with_both": sum(f["monetary"] and f["non_funding"] for f in flags),
            "records_with_label_like_values": sum(f["label_like"] for f in flags),
            "other_populated_values": {
                v: n for v, n in scholarships.items()
                if not any(scholarship_flags(v).values())
            },
        },
        "missing_or_empty_by_field": missing,
        "duplicate_project_ids": {key: count for key, count in ids.items() if count > 1},
        "suspicious_record_count": len(suspicious),
        "suspicious_reason_counts": dict(Counter(reason for r in suspicious for reason in r["reasons"])),
        "suspicious_records": suspicious,
        "manual_sample_rows": [index for index, _ in samples],
    }
    return report, [record for _, record in samples]


def main() -> int:
    try:
        rows = json.loads(SOURCE.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("projects.json must contain a JSON array")
        report, samples = audit(rows)
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nManual inspection: {len(samples)} complete records (seed {RANDOM_SEED})")
    for record in samples:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\nAudit saved to: {REPORT}")
    return 1 if report["validation_failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
