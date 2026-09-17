import subprocess
from pathlib import Path

from utas_research_assistant.ingestion.parse_local_docs import parse_directory
from utas_research_assistant.retrieval.bm25 import BM25Retriever
from utas_research_assistant.retrieval.corpus import build_corpus


def test_declared_private_document_ids_and_chunk_metadata(tmp_path: Path):
    for index in range(10):
        (tmp_path / f"doc_{index}.txt").write_text(
            f"Document ID: LOCAL-TEST-{index:02d}\nDocument type: User-created local support document\n"
            "Privacy: Private\n\nUnique local research rule content.", encoding="utf-8")
    documents, chunks, manifest = parse_directory(tmp_path)
    assert manifest["total_source_files"] == 10
    assert manifest["unique_document_ids"] == 10
    assert manifest["duplicate_document_ids"] == []
    assert {item.document_id for item in documents} == {f"LOCAL-TEST-{i:02d}" for i in range(10)}
    assert {item.document_id for item in chunks} == {f"LOCAL-TEST-{i:02d}" for i in range(10)}


def test_private_document_id_survives_bm25_metadata():
    row = {"chunk_id": "chunk-1", "document_id": "LOCAL-TEST-01", "title": "Private rule",
           "source_type": "local_document", "file_type": "txt", "local_filename": "rule.txt",
           "source_path": "/private/rule.txt", "category": "local_document",
           "text": "private applicant matching rule", "page_number": None, "chunk_index": 0, "word_count": 4}
    corpus = build_corpus([], [], [], [row])
    result = BM25Retriever(corpus).search("applicant matching")
    assert result[0]["document_id"] == "LOCAL-TEST-01"
    assert result[0]["item_type"] == "local_document"


def test_private_paths_are_ignored_by_git():
    root = Path(__file__).resolve().parents[1]
    check = subprocess.run(["git", "check-ignore", "-q", "data/raw/private/assignment_docs/example.txt"], cwd=root)
    assert check.returncode == 0
