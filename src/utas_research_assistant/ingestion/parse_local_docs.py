"""Offline parser for user-authored local documents.

Files are deliberately kept separate from the official UTAS general-document
pipeline and are labelled ``local_document`` throughout retrieval.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Literal
from xml.etree import ElementTree
from zipfile import ZipFile

from bs4 import BeautifulSoup
import pymupdf
from pydantic import BaseModel


SUPPORTED_EXTENSIONS = {".html", ".htm", ".txt", ".pdf", ".docx"}


class LocalDocumentRecord(BaseModel):
    document_id: str
    title: str
    source_type: Literal["local_document"]
    file_type: str
    local_filename: str
    source_path: str
    category: Literal["local_document"] = "local_document"
    text: str
    page_number: int | None = None
    parsed_at: datetime


class LocalChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source_type: Literal["local_document"]
    file_type: str
    local_filename: str
    source_path: str
    category: Literal["local_document"] = "local_document"
    text: str
    page_number: int | None = None
    chunk_index: int
    word_count: int


def _clean(text: str) -> str:
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _header_value(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None


def _content_without_metadata(text: str) -> str:
    """Keep required headers as metadata while indexing meaningful content."""
    lines = text.splitlines()
    header_keys = ("Document ID:", "Document type:", "Privacy:", "Status:",
                   "Official source:", "Note:")
    count = 0
    for line in lines:
        if not line.strip():
            if count >= 2:
                break
            continue
        if any(line.strip().startswith(key) for key in header_keys):
            count += 1
        else:
            break
    if count >= 2 and lines:
        while lines and lines[0].strip():
            lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
        return "\n".join(lines)
    return text


def _html_text(raw: str, fallback: str) -> tuple[str, str]:
    soup = BeautifulSoup(raw, "html.parser")
    title_node = soup.find("title") or soup.find("h1")
    title = " ".join(title_node.stripped_strings) if title_node else fallback
    for node in soup.select("script,style,noscript,nav,footer,aside,form,header"):
        node.decompose()
    main = soup.select_one("main") or soup.find("article") or soup.body or soup
    for node in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "section", "div"]):
        node.insert_before("\n\n"); node.insert_after("\n\n")
    return title.strip() or fallback, _clean(main.get_text("\n"))


def _docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = []
    for paragraph in root.iter(namespace + "p"):
        value = "".join(node.text or "" for node in paragraph.iter(namespace + "t"))
        if value.strip():
            paragraphs.append(value)
    return _clean("\n\n".join(paragraphs))


def _split_text(text: str, target: int = 420, maximum: int = 500, overlap: int = 60) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks, current = [], []
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > maximum:
            paragraphs_for_unit = [" ".join(words[i:i + maximum]) for i in range(0, len(words), maximum)]
        else:
            paragraphs_for_unit = [paragraph]
        for unit in paragraphs_for_unit:
            if current and len(" ".join(current + [unit]).split()) > maximum:
                chunks.append("\n\n".join(current))
                tail = " ".join("\n\n".join(current).split()[-overlap:])
                current = [tail] if tail else []
            current.append(unit)
            if len(" ".join(current).split()) >= target:
                chunks.append("\n\n".join(current))
                tail = " ".join("\n\n".join(current).split()[-overlap:])
                current = [tail] if tail else []
    if current:
        value = "\n\n".join(current)
        if not chunks or value != chunks[-1]:
            chunks.append(value)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def parse_directory(root: Path) -> tuple[list[LocalDocumentRecord], list[LocalChunkRecord], dict]:
    if not root.is_dir():
        raise FileNotFoundError(f"Local document directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS)
    parsed_at = datetime.now(timezone.utc)
    documents, chunks, failures = [], [], []
    for path in files:
        relative = path.relative_to(root).as_posix()
        try:
            suffix = path.suffix.casefold()
            title = path.stem.replace("_", " ")
            declared_document_id = None
            pages: list[tuple[int | None, str]]
            if suffix in {".html", ".htm"}:
                title, text = _html_text(path.read_text(encoding="utf-8"), title); pages = [(None, text)]
            elif suffix == ".txt":
                raw_text = path.read_text(encoding="utf-8")
                declared_document_id = _header_value(raw_text, "Document ID")
                pages = [(None, _clean(_content_without_metadata(raw_text)))]
            elif suffix == ".docx":
                pages = [(None, _docx_text(path))]
            else:
                with pymupdf.open(path) as pdf:
                    title = (pdf.metadata or {}).get("title") or title
                    pages = [(page.number + 1, _clean(page.get_text("text", sort=True))) for page in pdf]
            for page_number, text in pages:
                if not text:
                    continue
                identity = f"{relative}\0{page_number}"
                # User-authored assignment documents may provide a stable
                # internal identifier. Fall back to a deterministic filename
                # hash for ordinary local notes.
                declared_id = declared_document_id or _header_value(text, "Document ID")
                document_id = declared_id or ("local-" + hashlib.sha256(identity.encode()).hexdigest())
                document = LocalDocumentRecord(document_id=document_id, title=title, source_type="local_document",
                    file_type=suffix[1:], local_filename=relative, source_path=str(path), text=text,
                    page_number=page_number, parsed_at=parsed_at)
                documents.append(document)
                for index, chunk_text in enumerate(_split_text(text)):
                    chunk_id = hashlib.sha256(f"{document_id}\0{index}\0{chunk_text}".encode()).hexdigest()
                    chunks.append(LocalChunkRecord(chunk_id=chunk_id, document_id=document_id, title=title,
                        source_type="local_document", file_type=suffix[1:], local_filename=relative,
                        source_path=str(path), text=chunk_text, page_number=page_number,
                        chunk_index=index, word_count=len(chunk_text.split())))
        except (OSError, ValueError, RuntimeError, KeyError, ElementTree.ParseError) as exc:
            failures.append({"local_filename": relative, "error": str(exc)})
    manifest = {
        "source_folder": str(root), "total_source_files": len(files),
        "total_documents": len(documents), "total_chunks": len(chunks),
        "unique_document_ids": len({item.document_id for item in documents}),
        "duplicate_document_ids": sorted({item.document_id for item in documents
                                           if sum(other.document_id == item.document_id for other in documents) > 1}),
        "counts_by_file_type": dict(Counter(item.file_type for item in documents)),
        "empty_or_failed_files": failures,
        "total_extracted_characters": sum(len(item.text) for item in documents),
        "parsed_at": parsed_at.isoformat(),
    }
    return documents, chunks, manifest
