from pathlib import Path

from utas_research_assistant.ingestion.parse_local_docs import parse_directory
from utas_research_assistant.retrieval.corpus import build_corpus


def test_local_folder_discovery_and_metadata(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("Local research notes\n\nA useful paragraph.", encoding="utf-8")
    (tmp_path / "guide.html").write_text("<html><title>Local guide</title><main><h1>Heading</h1><p>HTML content.</p></main></html>", encoding="utf-8")
    documents, chunks, manifest = parse_directory(tmp_path)
    assert manifest["total_source_files"] == 2
    assert {item.file_type for item in documents} == {"txt", "html"}
    assert all(item.source_type == "local_document" and item.category == "local_document" for item in documents)
    assert all(item.document_id.startswith("local-") for item in chunks)
    assert all(item.local_filename in {"notes.txt", "guide.html"} for item in chunks)


def test_local_docx_parser_and_corpus_visibility(tmp_path: Path):
    # A small synthetic DOCX package avoids depending on python-docx.
    import zipfile
    xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>DOCX notes</w:t></w:r></w:p></w:body></w:document>'
    with zipfile.ZipFile(tmp_path / "notes.docx", "w") as archive:
        archive.writestr("word/document.xml", xml)
    _, chunks, _ = parse_directory(tmp_path)
    corpus = build_corpus([], [], [], [item.model_dump(mode="json") for item in chunks])
    assert chunks[0].file_type == "docx"
    assert corpus.items[0].item_type == "local_document"
    assert corpus.items[0].metadata["local_filename"] == "notes.docx"
