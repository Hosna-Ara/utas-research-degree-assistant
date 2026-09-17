"""Offline ingestion of manually saved general-knowledge documents."""

from collections import Counter
from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Literal
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Comment
import pymupdf
from pydantic import BaseModel, HttpUrl

Category = Literal[
    "research_degree_overview", "entry_requirements", "how_to_apply",
    "scholarships_and_fees", "faq", "other",
]
SUPPORTED_EXTENSIONS = {".html", ".htm", ".txt", ".pdf"}
CATEGORY_ALIASES = {
    "research_degree_overview": "research_degree_overview",
    "research_degrees_overview": "research_degree_overview",
    "entry_requirements": "entry_requirements",
    "how_to_apply": "how_to_apply",
    "scholarships_and_fees": "scholarships_and_fees",
    "research_degree_faq": "faq",
    "faq": "faq",
    "other": "other",
}


class GeneralDocumentRecord(BaseModel):
    document_id: str
    title: str
    source_type: Literal["html", "txt", "pdf"]
    local_filename: str
    source_url: HttpUrl | None = None
    category: Category
    text: str
    page_number: int | None = None
    fetched_or_saved_date: date | None = None
    parsed_at: datetime


def clean_text(value: str) -> str:
    """Clean line whitespace while retaining paragraph and page text boundaries."""
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in value.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def category_for(path: Path) -> Category:
    """Use a category folder or descriptive filename; unknown names stay 'other'."""
    for name in (*reversed(path.parts[:-1]), path.stem):
        normalized = re.sub(r"[\s-]+", "_", name.casefold())
        normalized = re.sub(r"_?\d{4}_\d{2}_\d{2}$", "", normalized)
        if normalized in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[normalized]
    return "other"


def clean_web_text(value: str) -> str:
    """Remove standalone UI labels, never social-media names inside prose."""
    labels = {"share this", "bluesky", "twitter", "facebook", "linkedin"}
    lines = []
    for line in value.splitlines():
        normalized = " ".join(line.split()).casefold()
        if normalized in labels or re.fullmatch(r"\|(?:\s*\|)*", normalized):
            continue
        # Saved accessibility labels sometimes repeat a link's visible name.
        if any(normalized == f"{label} {label}" for label in labels):
            continue
        lines.append(line)
    return clean_text("\n".join(lines))


def extract_html(html: str, fallback_title: str) -> tuple[str, str, str | None]:
    """Retain readable blocks and headings, including collapsed FAQ answers."""
    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.find_all(string=lambda node: isinstance(node, Comment)):
        comment.extract()
    title_node = soup.find("title") or soup.find("h1")
    title = " ".join(title_node.stripped_strings) if title_node else fallback_title
    canonical = soup.select_one('link[rel="canonical"]')
    og_url = soup.select_one('meta[property="og:url"]')
    source_url = canonical.get("href") if canonical else og_url.get("content") if og_url else None
    if source_url:
        source_url = str(source_url).strip()
        try:
            parts = urlsplit(source_url)
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                source_url = None
        except ValueError:
            source_url = None

    for node in soup.select(
        "script, style, noscript, template, head, nav, footer, aside, form, "
        "[role=navigation], [role=contentinfo], .sidebar-menu, .breadcrumbs-wrapper, "
        ".backtotop, .backtotop--padding, #onetrust-banner-sdk, #onetrust-consent-sdk, "
        ".cookie-banner, .cookie-consent, #cookie-banner, #cookie-consent, "
        ".social-media-sharing, .sharethis, [role=banner]"
    ):
        node.decompose()
    main = soup.select_one("main") or soup.select_one('[role="main"]')
    main = main or soup.find("article") or soup.body or soup
    # HTML source newlines do not imply paragraph breaks.
    for node in list(main.find_all(string=True)):
        node.replace_with(re.sub(r"\s+", " ", str(node)))
    # Block delimiters preserve paragraphs; inline emphasis stays within its paragraph.
    for node in main.find_all([
        "h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "section",
        "article", "li", "ul", "ol", "blockquote", "pre", "tr", "dt", "dd",
    ]):
        node.insert_before("\n\n")
        node.insert_after("\n\n")
    for node in main.find_all("br"):
        node.replace_with("\n")
    for node in main.find_all(["th", "td"]):
        node.insert_after(" | ")
    return title.strip() or fallback_title, clean_web_text(main.get_text()), source_url


def parse_file(path: Path, root: Path, *, parsed_at: datetime) -> tuple[list[GeneralDocumentRecord], list[int]]:
    """Return nonempty records and any empty PDF page numbers (1-based)."""
    relative = path.relative_to(root)
    source_type = "html" if path.suffix.lower() in {".html", ".htm"} else path.suffix.lower()[1:]
    title = path.stem.replace("_", " ")
    source_url = None
    if source_type == "pdf":
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError("PDF is password protected")
            title = (pdf.metadata or {}).get("title") or title
            pages = [(page.number + 1, clean_text(page.get_text("text", sort=True))) for page in pdf]
    elif source_type == "html":
        title, content, source_url = extract_html(path.read_text(encoding="utf-8"), title)
        pages = [(None, content)]
    elif source_type == "txt":
        pages = [(None, clean_text(path.read_text(encoding="utf-8")))]
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    records = []
    empty_pages = []
    for page_number, content in pages:
        if not content:
            if page_number is not None:
                empty_pages.append(page_number)
            continue
        identity = f"{relative.as_posix()}\0{page_number}"
        records.append(GeneralDocumentRecord(
            document_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            title=title.strip(), source_type=source_type,
            local_filename=relative.as_posix(), source_url=source_url,
            category=category_for(relative), text=content, page_number=page_number,
            parsed_at=parsed_at,
        ))
    return records, empty_pages


def parse_directory(root: Path) -> tuple[list[GeneralDocumentRecord], dict]:
    """Process each local source independently; report failures without losing others."""
    if not root.is_dir():
        raise FileNotFoundError(f"Local source directory does not exist: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    parsed_at = datetime.now(timezone.utc)
    records, empty_files, failed_files, empty_pdf_pages = [], [], [], []
    for path in files:
        name = path.relative_to(root).as_posix()
        try:
            parsed, empty_pages = parse_file(path, root, parsed_at=parsed_at)
        except (OSError, ValueError, RuntimeError) as exc:
            failed_files.append({"local_filename": name, "error": str(exc)})
            continue
        records.extend(parsed)
        if not parsed:
            empty_files.append(name)
        if empty_pages:
            empty_pdf_pages.append({"local_filename": name, "page_numbers": empty_pages})
    manifest = {
        "parsed_at": parsed_at.isoformat(),
        "total_source_files": len(files),
        "total_document_records": len(records),
        "counts_by_category": dict(Counter(r.category for r in records)),
        "counts_by_file_type": dict(Counter(p.suffix.lower()[1:] for p in files)),
        "empty_files": empty_files,
        "failed_files": failed_files,
        "empty_pdf_pages": empty_pdf_pages,
        "total_extracted_characters": sum(len(r.text) for r in records),
    }
    return records, manifest
