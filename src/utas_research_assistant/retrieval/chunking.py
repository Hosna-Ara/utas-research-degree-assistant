"""Paragraph-aware chunks of general documents, with deterministic identifiers."""

from collections import Counter
import hashlib
import re
from typing import Literal

from pydantic import BaseModel, HttpUrl

from utas_research_assistant.ingestion.parse_general_docs import Category, GeneralDocumentRecord

TARGET_WORDS = 400
MIN_WORDS = 250
MAX_WORDS = 500
OVERLAP_WORDS = 70


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    category: Category
    text: str
    source_type: Literal["html", "txt", "pdf"]
    source_url: HttpUrl | None
    local_filename: str
    page_number: int | None
    chunk_index: int
    word_count: int


def word_count(text: str) -> int:
    return len(text.split())


def is_heading(paragraph: str) -> bool:
    """Saved text has no heading tags; conservatively recognize short heading lines."""
    return (
        "\n" not in paragraph
        and word_count(paragraph) <= 20
        and not paragraph.endswith((".", "!", ";"))
    )


def paragraph_units(text: str) -> list[str]:
    """Attach likely headings to following text; split only oversized units."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    groups, pending = [], []
    for paragraph in paragraphs:
        pending.append(paragraph)
        if not is_heading(paragraph):
            groups.append("\n\n".join(pending))
            pending = []
    if pending:
        groups.append("\n\n".join(pending))

    units = []
    # Reserve space for overlap in the next chunk. Fallback preserves source wording.
    limit = MAX_WORDS - OVERLAP_WORDS
    for group in groups:
        words = list(re.finditer(r"\S+", group))
        for start in range(0, len(words), limit):
            end = min(start + limit, len(words))
            units.append(group[words[start].start():words[end - 1].end()])
    return units


def overlap_tail(text: str) -> str:
    """Repeat the last 70 words, retaining their internal paragraph boundaries."""
    words = list(re.finditer(r"\S+", text))
    return text[words[max(0, len(words) - OVERLAP_WORDS)].start():] if words else ""


def chunk_document(document: GeneralDocumentRecord) -> list[ChunkRecord]:
    chunks, current = [], ""
    for unit in paragraph_units(document.text):
        size = word_count(current)
        combined_size = size + word_count(unit)
        if current and (
            combined_size > MAX_WORDS
            or (size >= MIN_WORDS and abs(size - TARGET_WORDS) <= abs(combined_size - TARGET_WORDS))
        ):
            chunks.append(current)
            current = overlap_tail(current)
        current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)

    metadata = document.model_dump(include={
        "document_id", "title", "category", "source_type", "source_url",
        "local_filename", "page_number",
    })
    return [
        ChunkRecord(
            **metadata,
            chunk_id=hashlib.sha256(
                f"{document.document_id}\0{index}\0{text}".encode("utf-8")
            ).hexdigest(),
            text=text, chunk_index=index, word_count=word_count(text),
        )
        for index, text in enumerate(chunks)
    ]


def build_chunks(documents: list[GeneralDocumentRecord]) -> tuple[list[ChunkRecord], dict]:
    if len({d.document_id for d in documents}) != len(documents):
        raise ValueError("Duplicate document IDs; chunk identifiers would be ambiguous")
    chunks, empty = [], []
    for document in documents:
        document_chunks = chunk_document(document)
        chunks.extend(document_chunks)
        if not document_chunks:
            empty.append({"document_id": document.document_id, "local_filename": document.local_filename})
    sizes = [chunk.word_count for chunk in chunks]
    manifest = {
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "chunks_by_category": dict(Counter(chunk.category for chunk in chunks)),
        "min_word_count": min(sizes, default=0),
        "average_word_count": round(sum(sizes) / len(sizes), 2) if sizes else 0,
        "max_word_count": max(sizes, default=0),
        "documents_producing_zero_chunks": empty,
        "settings": {
            "target_words": TARGET_WORDS, "min_words": MIN_WORDS,
            "max_words": MAX_WORDS, "overlap_words": OVERLAP_WORDS,
        },
    }
    return chunks, manifest
