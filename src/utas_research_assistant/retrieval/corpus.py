"""Unified local corpus; only exact general-chunk text is deduplicated."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Literal

from utas_research_assistant.retrieval.chunking import ChunkRecord
from utas_research_assistant.retrieval.project_documents import ProjectDocument
from utas_research_assistant.retrieval.supervisor_documents import SupervisorDocument
from utas_research_assistant.ingestion.parse_local_docs import LocalChunkRecord


def tokenize(text: str) -> list[str]:
    """Lowercase Unicode words/numbers; punctuation and underscores are separators."""
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


@dataclass
class CorpusItem:
    item_type: Literal["general_chunk", "research_project", "supervisor_profile", "local_document"]
    text: str
    tokens: list[str]
    metadata: dict
    provenance: list[dict]


@dataclass
class Corpus:
    items: list[CorpusItem]
    raw_item_count: int

    @property
    def summary(self) -> dict:
        return {
            "raw_item_count": self.raw_item_count,
            "indexed_item_count": len(self.items),
            "duplicates_removed": self.raw_item_count - len(self.items),
        }


def build_corpus(general_chunks: list[dict], project_documents: list[dict],
                 supervisor_documents: list[dict] | None = None,
                 local_documents: list[dict] | None = None) -> Corpus:
    items, general_by_text = [], {}
    profile_lookup = {}
    for profile in supervisor_documents or []:
        name = profile.get("canonical_name") or profile.get("title") or ""
        key = re.sub(r"^(?:associate\s+professor|assoc\.?\s+prof|professor|prof\.?|doctor|dr\.?)\s+", "", " ".join(str(name).casefold().split()))
        if key:
            profile_lookup[key] = profile
    for item_type, rows, model in (
        ("general_chunk", general_chunks, ChunkRecord),
        ("research_project", project_documents, ProjectDocument),
        ("supervisor_profile", supervisor_documents or [], SupervisorDocument),
        ("local_document", local_documents or [], LocalChunkRecord),
    ):
        for index, row in enumerate(rows, 1):
            try:
                validated = model.model_validate(row).model_dump(mode="json")
            except ValueError as exc:
                raise ValueError(f"Invalid {item_type} row {index}: {exc}") from exc
            text = validated.pop("text")
            # Keep additional source metadata too; no input dictionary is mutated.
            metadata = {**{k: v for k, v in row.items() if k != "text"}, **validated}
            if item_type == "research_project":
                supervisor = metadata.get("primary_supervisor") or ""
                key = re.sub(r"^(?:associate\s+professor|assoc\.?\s+prof|professor|prof\.?|doctor|dr\.?)\s+", "", " ".join(str(supervisor).casefold().split()))
                profile = profile_lookup.get(key)
                if profile:
                    metadata["supervisor_profile"] = {
                        field: profile.get(field) for field in (
                            "canonical_name", "title", "school", "research_fields",
                            "discovery_profile_id", "discovery_url", "source_url",
                            "orcid", "google_scholar_url",
                        ) if profile.get(field)
                    }
            provenance = {"item_type": item_type, **metadata}
            if item_type == "general_chunk" and text in general_by_text:
                general_by_text[text].provenance.append(provenance)
                continue
            tokens = tokenize(text)
            if not tokens:
                raise ValueError(f"No indexable text in {item_type} row {index}")
            item = CorpusItem(item_type, text, tokens, metadata, [provenance])
            items.append(item)
            if item_type == "general_chunk":
                general_by_text[text] = item
    return Corpus(items, len(general_chunks) + len(project_documents) + len(supervisor_documents or []) + len(local_documents or []))


def load_corpus(processed_dir: Path) -> Corpus:
    def read_array(filename: str) -> list[dict]:
        rows = json.loads((processed_dir / filename).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{filename} must contain a JSON array")
        return rows

    supervisor_path = processed_dir / "supervisor_documents.json"
    supervisors = read_array("supervisor_documents.json") if supervisor_path.exists() else []
    local_path = processed_dir / "local_chunks.json"
    local = read_array("local_chunks.json") if local_path.exists() else []
    return build_corpus(read_array("general_chunks.json"), read_array("project_documents.json"), supervisors, local)
