from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import pytest

from utas_research_assistant.ingestion.parse_general_docs import (
    category_for, clean_web_text, extract_html, parse_directory, parse_file,
)

PARSED_AT = datetime(2026, 9, 15, tzinfo=timezone.utc)


def test_standalone_web_boilerplate_and_separators():
    original = (
        "Entry requirements\n\nShare this\n\nbluesky Bluesky\nTwitter\nFacebook\nLinkedIn\n"
        "|\n\nScore | Requirement\n\n| |\n\n"
        "Research on Twitter and Facebook remains relevant.\n\nScholarships are available."
    )
    cleaned = clean_web_text(original)
    assert cleaned == (
        "Entry requirements\n\nScore | Requirement\n\n"
        "Research on Twitter and Facebook remains relevant.\n\nScholarships are available."
    )


def test_saved_sharethis_widget_and_table_content():
    _, content, _ = extract_html(
        '<main><section class="sharethis"><span>Share this</span><a>Bluesky</a></section>'
        '<h2>Language requirements</h2><table><tr><td><p>IELTS</p></td>'
        '<td><p>7.0</p></td></tr></table>'
        '<p>Study LinkedIn networks.</p></main>', "Guide",
    )
    assert "Share this" not in content
    assert "Bluesky" not in content
    assert not any(line.strip() == "|" for line in content.splitlines())
    assert all(word in content for word in ("Language requirements", "IELTS", "7.0", "Study LinkedIn networks."))


def test_html_readable_content_and_metadata():
    html = """<html><head><title>Degree guide</title>
    <link rel="canonical" href="https://www.utas.edu.au/research/degrees"></head>
    <body><header>Global navigation</header><main><nav>Breadcrumbs</nav>
    <!-- Internal implementation comment -->
    <div class="social-media-sharing">Share this page</div>
    <div class="sidebar-menu">Sidebar links</div><h1>Research degrees</h1>
    <p>First <strong>important</strong> paragraph.</p><p>Second paragraph.</p>
    <h2>Questions</h2><div hidden><p>Collapsed FAQ answer.</p></div>
    <div class="cookie-banner">Accept cookies</div><footer>Footer links</footer>
    <script>fetch('never execute this')</script></main></body></html>"""
    title, content, url = extract_html(html, "fallback")
    assert title == "Degree guide"
    assert url == "https://www.utas.edu.au/research/degrees"
    assert "Research degrees\n\nFirst important paragraph.\n\nSecond paragraph." in content
    assert "Questions\n\nCollapsed FAQ answer." in content
    for unwanted in ("navigation", "Breadcrumbs", "Sidebar", "cookies", "Footer", "fetch", "implementation", "Share this"):
        assert unwanted not in content


def test_html_fallback_and_invalid_url():
    title, content, url = extract_html(
        '<meta property="og:url" content="file:///private/snapshot.html"><body><p>Useful text.</p></body>',
        "Saved guide",
    )
    assert (title, content, url) == ("Saved guide", "Useful text.", None)


@pytest.mark.parametrize("name, expected", [
    ("research_degrees_overview.html", "research_degree_overview"),
    ("entry_requirements_2026-09-15.htm", "entry_requirements"),
    ("faq/unknown.pdf", "faq"),
    ("how_to_apply.txt", "how_to_apply"),
    ("scholarships_and_fees.html", "scholarships_and_fees"),
    ("research_degree_faq.html", "faq"),
    ("notes.txt", "other"),
])
def test_categories(name, expected):
    assert category_for(Path(name)) == expected


def test_utf8_text_and_stable_ids(tmp_path):
    path = tmp_path / "faq.txt"
    path.write_text("Heading\n\n  Café   question.\n\nAnswer.\n", encoding="utf-8")
    records, empty_pages = parse_file(path, tmp_path, parsed_at=PARSED_AT)
    record, = records
    assert record.text == "Heading\n\nCafé question.\n\nAnswer."
    assert record.source_type == "txt"
    assert record.category == "faq"
    assert record.local_filename == "faq.txt"
    assert record.page_number is None
    assert record.source_url is None
    assert record.fetched_or_saved_date is None
    assert record.parsed_at == PARSED_AT
    assert empty_pages == []
    again, _ = parse_file(path, tmp_path, parsed_at=datetime.now(timezone.utc))
    assert again[0].document_id == record.document_id


def test_pdf_pages_and_blank_page_metadata(tmp_path):
    path = tmp_path / "entry_requirements.pdf"
    with pymupdf.open() as pdf:
        pdf.set_metadata({"title": "Entry guide"})
        pdf.new_page().insert_text((72, 72), "Page one: requirements")
        pdf.new_page()
        pdf.new_page().insert_text((72, 72), "Page three: qualifications")
        pdf.save(path)
    records, empty_pages = parse_file(path, tmp_path, parsed_at=PARSED_AT)
    assert [r.page_number for r in records] == [1, 3]
    assert empty_pages == [2]
    assert all(r.title == "Entry guide" and r.source_type == "pdf" for r in records)
    assert all(r.category == "entry_requirements" for r in records)
    assert "requirements" in records[0].text
    assert "qualifications" in records[1].text
    assert records[0].document_id != records[1].document_id
    _, manifest = parse_directory(tmp_path)
    assert manifest["empty_pdf_pages"] == [{"local_filename": path.name, "page_numbers": [2]}]


def test_directory_manifest_and_individual_failures(tmp_path):
    (tmp_path / "faq.htm").write_text("<main><h1>FAQ</h1><p>An answer.</p></main>")
    (tmp_path / "other.txt").write_text("Useful notes", encoding="utf-8")
    (tmp_path / "empty.txt").write_text(" \n\t")
    (tmp_path / "broken.pdf").write_bytes(b"not a PDF")
    (tmp_path / "invalid.txt").write_bytes(b"\xff")
    (tmp_path / "ignored.json").write_text("{}")
    records, manifest = parse_directory(tmp_path)
    assert len(records) == 2
    assert manifest["total_source_files"] == 5
    assert manifest["total_document_records"] == 2
    assert manifest["counts_by_category"] == {"faq": 1, "other": 1}
    assert manifest["counts_by_file_type"] == {"pdf": 1, "txt": 3, "htm": 1}
    assert manifest["empty_files"] == ["empty.txt"]
    assert {f["local_filename"] for f in manifest["failed_files"]} == {"broken.pdf", "invalid.txt"}
    assert manifest["total_extracted_characters"] == sum(len(r.text) for r in records)


def test_missing_or_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="source directory"):
        parse_directory(tmp_path / "missing")
    records, manifest = parse_directory(tmp_path)
    assert records == []
    assert manifest["total_source_files"] == manifest["total_extracted_characters"] == 0
