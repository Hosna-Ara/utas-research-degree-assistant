from datetime import datetime, timezone
from html import escape
import json

import pytest

from utas_research_assistant.ingestion.parse_projects import (
    clean_text,
    parse_projects,
    project_link,
    split_types,
)

FETCHED_AT = datetime(2026, 9, 14, tzinfo=timezone.utc)


@pytest.fixture
def card():
    def make(href="/research/degrees/available-projects?id=12167", title="A project", details=""):
        return f"""
        <li class="hdr-project__card">
          <div class="hdr-project__card--status">
            <div class="priority">Priority project</div>
            <div class="open">Applications open</div>
          </div>
          <a class="hdr-project__card--title" href="{escape(href)}">{title}</a>
          <div class="project__card--category">Marine and Antarctic</div>
          <div class="hdr-project__card--text">Study <em>marine</em> life &amp; habitats.</div>
          {details}
        </li>"""
    return make


def detail(label, value):
    return f"""<div class="hdr-project__card--details--item">
    <span class="hdr-project__card--details--label">{label}</span>{value}</div>"""


def listing(*cards):
    return '<ul class="hdr-project__listing">' + "".join(cards) + "</ul>"


def test_text_helpers_preserve_wording():
    assert clean_text(" Up to\n $60,000\xa0pa ") == "Up to $60,000 pa"
    assert split_types("PhD and Master by Research") == ["PhD", "Master by Research"]
    assert split_types("") == []


@pytest.mark.parametrize("href", [
    "?id=12167", "/research/degrees/available-projects?id=12167",
    "https://www.utas.edu.au/research/degrees/available-projects?x=1&id=12167",
])
def test_project_link_id_and_relative_urls(href):
    project_id, url = project_link(href)
    assert project_id == "12167"
    assert url.startswith("https://www.utas.edu.au/research/degrees/available-projects?")


@pytest.mark.parametrize("href", ["", "#", "javascript:alert(1)", "https://example.com/?id=1"])
def test_invalid_links_rejected(href):
    with pytest.raises(ValueError):
        project_link(href)


def test_card_fields_and_embedded_location(card):
    details = "".join([
        detail("Degree type", "PhD and Master by Research"),
        detail("Student type", "Domestic and International"),
        detail("Scholarship", " Up to <b>$60,000</b> pa "),
        detail("Primary supervisor", '<a href="/staff/example">Doctor A Example</a>'),
        detail("Closing date ", "1 October 2026"),
    ])
    metadata = json.dumps([{"id": 12167, "locations": ["Hobart", "Launceston"]}])
    html = listing(card(details=details)) + f"<script>const hdrData = {metadata};</script>"
    record, = parse_projects(html, fetched_at=FETCHED_AT)
    assert record.project_id == "12167"
    assert record.status == "Applications open"
    assert record.degree_types == ["PhD", "Master by Research"]
    assert record.student_types == ["Domestic", "International"]
    assert record.scholarship_text == "Up to $60,000 pa"
    assert record.primary_supervisor == "Doctor A Example"
    assert record.closing_date == "1 October 2026"
    assert record.location == "Hobart; Launceston"
    assert record.research_categories == ["Marine and Antarctic"]
    assert record.description == "Study marine life & habitats."
    assert record.eligibility is None
    assert record.selection_criteria is None
    assert record.fetched_at == FETCHED_AT


def test_closed_card_does_not_invent_date_or_missing_fields(card):
    html = listing(card(details=detail("Closed", "Under Assessment")))
    html = html.replace('class="open">Applications open', 'class="closed">Under assessment')
    record, = parse_projects(html, fetched_at=FETCHED_AT)
    assert record.status == "Under assessment"
    assert record.closing_date is None
    assert record.location is None
    assert record.scholarship_text is None
    assert record.primary_supervisor is None
    assert record.degree_types == []
    assert record.student_types == []


def test_deduplication_by_id_and_fallback_url(card, caplog):
    html = listing(
        card(), card("?id=12167&tracking=example", title="Duplicate"),
        card("/research/degrees/available-projects/example"),
        card("https://www.utas.edu.au/research/degrees/available-projects/example"),
        card("/research/degrees/available-projects/another"),
    )
    records = parse_projects(html, fetched_at=FETCHED_AT)
    assert len(records) == 3
    assert [r.project_id for r in records] == ["12167", "", ""]
    assert records[0].title == "A project"
    assert "Duplicate card" in caplog.text
    assert "using URL for deduplication" in caplog.text


def test_malformed_and_validation_failures_are_reported(card, caplog):
    html = listing(card(title=""), card(""), card("?id=" + "x" * 2100), card())
    records = parse_projects(html, fetched_at=FETCHED_AT)
    assert len(records) == 1
    assert "Malformed card 1" in caplog.text
    assert "Card 3 validation failed" in caplog.text
    assert "Skipped 3 of 4 listing cards" in caplog.text


def test_bad_embedded_metadata_warns_but_keeps_cards(card, caplog):
    html = listing(card()) + '<script>const hdrData = not_json;</script>'
    record, = parse_projects(html, fetched_at=FETCHED_AT)
    assert record.location is None
    assert "Cannot read embedded listing locations" in caplog.text


def test_empty_or_entirely_invalid_listing_fails(card):
    with pytest.raises(ValueError, match="No UTAS project listing cards"):
        parse_projects("<html></html>", fetched_at=FETCHED_AT)
    with pytest.raises(ValueError, match="All listing cards failed"):
        parse_projects(listing(card(title="")), fetched_at=FETCHED_AT)
