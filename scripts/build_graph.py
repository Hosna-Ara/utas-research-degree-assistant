"""Build the local UTAS project RDF graph from processed project documents."""

import json
from datetime import datetime, timezone
from pathlib import Path

from utas_research_assistant.graph.builder import build_graph, graph_statistics
from utas_research_assistant.retrieval.project_documents import ProjectDocument

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/project_documents.json"
SUPERVISOR_SOURCE = ROOT / "data/processed/supervisor_profiles.json"
OUTPUT = ROOT / "data/processed/utas_research_graph.ttl"
MANIFEST = ROOT / "data/processed/graph_manifest.json"


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"Project document source file not found: {SOURCE}")
    records = [ProjectDocument.model_validate(item) for item in json.loads(SOURCE.read_text(encoding="utf-8"))]
    supervisor_profiles = []
    if SUPERVISOR_SOURCE.exists():
        supervisor_profiles = json.loads(SUPERVISOR_SOURCE.read_text(encoding="utf-8"))
    graph = build_graph(records, supervisor_profiles)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=OUTPUT, format="turtle", encoding="utf-8")
    stats = graph_statistics(graph)
    manifest = {
        **stats,
        "source_file": str(SOURCE.relative_to(ROOT)),
        "supervisor_source_file": str(SUPERVISOR_SOURCE.relative_to(ROOT)) if SUPERVISOR_SOURCE.exists() else None,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Graph statistics:")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print(f"Turtle: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
