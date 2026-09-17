"""Build a public-UTAS-only runtime bundle; never copies local documents."""

import argparse
import json
from pathlib import Path
import shutil

from utas_research_assistant.deployment import bundle_manifest, validate_runtime_data
from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.semantic import build_embeddings

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = ("general_chunks.json", "project_documents.json", "supervisor_documents.json",
                "utas_research_graph.ttl", "graph_manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/utas-public-runtime-data")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not (source / "project_documents.json").exists():
        raise SystemExit(f"Public source data is incomplete: {source}")
    output.mkdir(parents=True, exist_ok=True)
    for old in output.iterdir():
        if old.is_file():
            old.unlink()
        elif old.is_dir():
            shutil.rmtree(old)
    for filename in PUBLIC_FILES:
        source_file = source / filename
        if not source_file.is_file():
            raise SystemExit(f"Required public artifact is missing: {source_file}")
        shutil.copy2(source_file, output / filename)
    # The copied directory contains no local_chunks.json, so this index is
    # independently generated from public projects/general/supervisor rows.
    metadata = build_embeddings(load_corpus(output), output)
    manifest = bundle_manifest(output)
    general = json.loads((output / "general_chunks.json").read_text())
    projects = json.loads((output / "project_documents.json").read_text())
    supervisors = json.loads((output / "supervisor_documents.json").read_text())
    target_path = source / "supervisor_targets.json"
    target_count = len(json.loads(target_path.read_text(encoding="utf-8"))) if target_path.exists() else len(supervisors)
    manifest.update({"projects": len(projects), "supervisors": len(supervisors),
                     "supervisors": target_count,
                     "resolved_supervisor_profiles": sum(bool(row.get("discovery_url")) for row in supervisors),
                     "general_chunks": len(general), "semantic_vectors": metadata["embedding_item_count"]})
    (output / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    validate_runtime_data(output, public_only=True)
    print(json.dumps({"output": str(output), "file_count": len(manifest["files"]),
                      "bytes": sum(item["bytes"] for item in manifest["files"]),
                      "projects": manifest["projects"], "supervisors": manifest["supervisors"],
                      "semantic_vectors": manifest["semantic_vectors"],
                      "private_local_documents_included": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
