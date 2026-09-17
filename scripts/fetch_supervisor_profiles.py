"""Resolve, fetch, normalize, and report UTAS Discover supervisor profiles."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from utas_research_assistant.ingestion.supervisor_profiles import (
    DISCOVER_API,
    SupervisorProfile,
    SupervisorTarget,
    candidate_profile_ids,
    ict_quality_gate,
    load_cached_profile,
    make_session,
    normalize_profile,
    normalize_name,
    resolve_target,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
RAW = ROOT / "data/raw/supervisors"
ICT_CATEGORY = "Information and Communication Technology"


def derive_targets(path: Path = PROCESSED / "project_documents.json") -> list[SupervisorTarget]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, dict] = {}
    for row in rows:
        name = str(row.get("primary_supervisor") or "").strip()
        if not name:
            continue
        key = normalize_name(name)
        target = grouped.setdefault(key, {"name": name, "project_ids": set(), "categories": set()})
        target["project_ids"].add(str(row.get("project_id")))
        target["categories"].update(str(c) for c in row.get("research_categories", []) if c)
    return [SupervisorTarget(
        canonical_project_supervisor_name=value["name"], normalized_name=key,
        project_ids=sorted(value["project_ids"]), research_categories=sorted(value["categories"]),
        is_ict_supervisor=ICT_CATEGORY in value["categories"],
    ) for key, value in sorted(grouped.items())]


def _raw_path(profile_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", profile_id)
    return RAW / f"{safe}.json"


def build_document(profile: SupervisorProfile) -> dict:
    parts = [f"Supervisor: {profile.canonical_name}"]
    for label, value in (("Role", profile.title), ("School", profile.school), ("Sub-organisation", profile.sub_organisation), ("Bio", profile.bio), ("Research fields", ", ".join(profile.research_fields) if profile.research_fields else None), ("ORCID", profile.orcid), ("Google Scholar", profile.google_scholar_url), ("Positions", "; ".join(map(str, profile.positions)) if profile.positions else None), ("Degrees", "; ".join(map(str, profile.degrees)) if profile.degrees else None), ("Related advertised projects", ", ".join(profile.related_project_ids))):
        if value:
            parts.append(f"{label}: {value}")
    return {
        "document_id": f"supervisor-document-{profile.supervisor_id}",
        "supervisor_id": profile.supervisor_id,
        "text": "\n\n".join(parts),
        "title": profile.canonical_name,
        "source_url": str(profile.source_url),
        "discovery_profile_id": profile.discovery_profile_id,
        "related_project_ids": profile.related_project_ids,
        "related_research_categories": profile.related_research_categories,
        "is_ict_supervisor": profile.is_ict_supervisor,
        "canonical_name": profile.canonical_name,
        "school": profile.school,
        "research_fields": profile.research_fields,
        "orcid": profile.orcid,
        "google_scholar_url": profile.google_scholar_url,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="refetch validated cached profiles")
    parser.add_argument("--delay", type=float, default=1.2, help="delay between profile requests")
    args = parser.parse_args()
    targets = derive_targets()
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "supervisor_targets.json").write_text(json.dumps([t.model_dump() for t in targets], indent=2), encoding="utf-8")
    ict = [t for t in targets if t.is_ict_supervisor]
    print(f"Total unique supervisors: {len(targets)}")
    print(f"Unique ICT supervisors: {len(ict)}")
    print("ICT supervisor names:")
    for target in ict:
        print(f"- {target.canonical_project_supervisor_name}")

    session = make_session()
    profiles: list[SupervisorProfile] = []
    unresolved = []
    fetch_failures = []
    for index, target in enumerate(targets, start=1):
        cached = None
        if not args.refresh:
            for candidate in candidate_profile_ids(target.canonical_project_supervisor_name):
                cached = load_cached_profile(_raw_path(candidate), target)
                if cached:
                    break
        if cached:
            profiles.append(cached)
            continue
        result = resolve_target(session, target, delay=args.delay)
        if result.get("payload"):
            profile_id = result["profile_id"]
            raw_path = _raw_path(profile_id)
            raw_path.write_text(json.dumps(result["payload"], indent=2, ensure_ascii=False), encoding="utf-8")
            profile = normalize_profile(result["payload"], target, profile_id)
            profiles.append(profile)
        else:
            attempts = result.get("attempts", [])
            unresolved.append({
                "supervisor_name": target.canonical_project_supervisor_name,
                "related_project_ids": target.project_ids,
                "research_categories": target.research_categories,
                "attempted_profile_ids": [a["profile_id"] for a in attempts],
                "attempted_urls": [a["url"] for a in attempts],
                "failure_reason": result.get("failure_reason"),
                "suggested_next_action": "Check the Discover profile identifier manually and rerun with a validated raw JSON profile.",
                "is_ict_supervisor": target.is_ict_supervisor,
            })
            fetch_failures.extend(attempts)
        if index < len(targets):
            # resolve_target already spaces candidate requests; this keeps the
            # boundary between different people polite as well.
            import time
            time.sleep(max(0.0, args.delay))

    profiles.sort(key=lambda p: p.canonical_name.casefold())
    (PROCESSED / "supervisor_profiles.json").write_text(json.dumps([p.model_dump(mode="json") for p in profiles], indent=2, ensure_ascii=False), encoding="utf-8")
    with (PROCESSED / "supervisor_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["supervisor_id", "canonical_name", "discovery_profile_id", "school", "bio", "orcid", "is_ict_supervisor", "related_project_ids"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            row = profile.model_dump(mode="json")
            row["related_project_ids"] = "; ".join(profile.related_project_ids)
            writer.writerow({key: row.get(key) for key in fields})
    documents = [build_document(profile) for profile in profiles]
    (PROCESSED / "supervisor_documents.json").write_text(json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8")
    (PROCESSED / "supervisor_manual_resolution.json").write_text(json.dumps(unresolved, indent=2, ensure_ascii=False), encoding="utf-8")

    resolved_ict = sum(p.is_ict_supervisor for p in profiles)
    manifest = {
        "total_unique_supervisors": len(targets), "total_profiles_resolved": len(profiles),
        "total_profiles_unresolved": len(unresolved), "overall_resolution_rate": len(profiles) / len(targets) if targets else 0,
        "ict_unique_supervisors": len(ict), "ict_profiles_resolved": resolved_ict,
        "ict_profiles_unresolved": len(ict) - resolved_ict,
        "ict_resolution_rate": resolved_ict / len(ict) if ict else 1.0,
        "profiles_with_bio": sum(bool(p.bio) for p in profiles),
        "profiles_with_research_fields": sum(bool(p.research_fields) for p in profiles),
        "profiles_with_orcid": sum(bool(p.orcid) for p in profiles),
        "profiles_with_google_scholar": sum(bool(p.google_scholar_url) for p in profiles),
        "profiles_with_school": sum(bool(p.school) for p in profiles),
        "fetch_failures": fetch_failures, "unresolved_supervisors": unresolved,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (PROCESSED / "supervisor_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resolved profiles: {len(profiles)}")
    print(f"Unresolved profiles: {len(unresolved)}")
    print(f"ICT resolved: {resolved_ict}")
    print(f"ICT unresolved: {len(ict) - resolved_ict}")
    print(f"ICT success rate: {resolved_ict / len(ict) if ict else 1.0:.1%}")
    if ict_quality_gate(targets, profiles):
        print("ICT PROFILE QUALITY GATE: PASSED")
    else:
        print("ICT PROFILE QUALITY GATE: FAILED")
        print("Unresolved ICT supervisors:")
        for item in unresolved:
            if item.get("is_ict_supervisor"):
                print(f"- {item['supervisor_name']}: {', '.join(item['attempted_profile_ids'])}")
    return 0 if resolved_ict == len(ict) else 2


if __name__ == "__main__":
    raise SystemExit(main())
