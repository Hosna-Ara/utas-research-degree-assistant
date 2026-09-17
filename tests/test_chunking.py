from datetime import datetime, timezone

import pytest

from utas_research_assistant.ingestion.parse_general_docs import GeneralDocumentRecord
from utas_research_assistant.retrieval.chunking import build_chunks, chunk_document


def paragraph(prefix, size):
    return " ".join(f"{prefix}{i}" for i in range(size)) + "."


def document(text, document_id="doc-1"):
    return GeneralDocumentRecord(
        document_id=document_id, title="Entry guide", category="entry_requirements",
        source_type="pdf", local_filename="entry.pdf", page_number=3,
        source_url="https://www.utas.edu.au/research/degrees",
        parsed_at=datetime(2026, 9, 15, tzinfo=timezone.utc), text=text,
    )


def test_paragraph_splitting_keeps_heading_with_following_paragraph():
    paragraphs = [paragraph(f"p{i}-", 120) for i in range(5)]
    text = "\n\n".join(paragraphs[:3] + ["Entry requirements", paragraphs[3], paragraphs[4]])
    chunks = chunk_document(document(text))
    assert len(chunks) == 2
    assert chunks[0].text == "\n\n".join(paragraphs[:3])
    assert "Entry requirements\n\n" + paragraphs[3] in chunks[1].text
    assert all(not chunk.text.endswith("Entry requirements") for chunk in chunks)
    assert all(any(p in c.text for c in chunks) for p in paragraphs)


def test_overlap_and_complete_word_coverage():
    text = "\n\n".join(paragraph(f"p{i}-", 200) for i in range(10))
    chunks = chunk_document(document(text))
    reconstructed = chunks[0].text.split()
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.text.split()[-70:] == current.text.split()[:70]
        reconstructed.extend(current.text.split()[70:])
    assert reconstructed == text.split()
    assert all(250 <= c.word_count <= 500 for c in chunks[:-1])


def test_stable_ids_and_independent_document_boundaries():
    first = document(paragraph("a", 1500))
    second = document(paragraph("b", 1500), "doc-2")
    chunks, _ = build_chunks([first, second])
    reversed_chunks, _ = build_chunks([second, first])
    assert {c.chunk_id for c in chunks} == {c.chunk_id for c in reversed_chunks}
    assert chunk_document(first) == chunk_document(first)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all("b0" not in c.text for c in chunks if c.document_id == "doc-1")
    previous_last_id = chunk_document(first)[-1].chunk_id
    first.text += " Added wording."
    assert chunk_document(first)[-1].chunk_id != previous_last_id


def test_metadata_and_indices():
    source = document(paragraph("long", 1800))
    chunks = chunk_document(source)
    for index, chunk in enumerate(chunks):
        for field in ("document_id", "title", "category", "source_type", "source_url", "local_filename", "page_number"):
            assert getattr(chunk, field) == getattr(source, field)
        assert chunk.chunk_index == index
        assert chunk.word_count == len(chunk.text.split())
        assert chunk.word_count <= 500
    source.page_number = None
    source.source_url = None
    assert chunk_document(source)[0].page_number is None
    assert chunk_document(source)[0].source_url is None


def test_long_paragraph_fallback_loses_no_words():
    source = document(paragraph("long", 2000))
    chunks = chunk_document(source)
    words = chunks[0].text.split()
    for chunk in chunks[1:]:
        words.extend(chunk.text.split()[70:])
    assert words == source.text.split()
    assert all(c.word_count <= 500 for c in chunks)


def test_short_empty_documents_and_manifest():
    short = document("A short document.")
    empty = document(" \n\n ", "empty")
    chunks, manifest = build_chunks([short, empty])
    assert len(chunks) == 1
    assert manifest["total_documents"] == 2
    assert manifest["chunks_by_category"] == {"entry_requirements": 1}
    assert manifest["min_word_count"] == manifest["max_word_count"] == 3
    assert manifest["documents_producing_zero_chunks"] == [{"document_id": "empty", "local_filename": "entry.pdf"}]
    assert build_chunks([])[1]["average_word_count"] == 0
    with pytest.raises(ValueError, match="Duplicate document IDs"):
        build_chunks([short, short])
