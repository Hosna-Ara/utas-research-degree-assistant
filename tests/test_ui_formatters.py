from utas_research_assistant.ui.formatters import (
    metadata_value, official_source_url, official_supervisor_url, project_cards_from_sources, source_display_title,
)


def test_project_cards_preserve_metadata_from_provenance():
    cards = project_cards_from_sources([{
        "citation_id": "S1", "project_id": "12259", "title": "Phishing detection",
        "source_url": "https://www.utas.edu.au/research/degrees/available-projects?id=12259",
        "provenance": [{
            "item_type": "research_project", "project_id": "12259",
            "degree_types": ["PhD", "Master by Research"], "student_types": ["International"],
            "research_categories": ["Information and Communication Technology"],
            "scholarship_text": "$34,315 pa", "primary_supervisor": "Doctor Example",
        }],
    }])
    assert len(cards) == 1
    assert cards[0]["degree_types"] == ["PhD", "Master by Research"]
    assert cards[0]["scholarship_text"] == "$34,315 pa"
    assert cards[0]["source_url"].startswith("https://www.utas.edu.au/")


def test_cards_skip_missing_fields_and_deduplicate_project_ids():
    source = {"citation_id": "S1", "project_id": "42", "title": "Example"}
    assert len(project_cards_from_sources([source, source])) == 1
    assert metadata_value(None) is None
    assert metadata_value([]) is None
    assert metadata_value(["PhD", "MRes"]) == "PhD, MRes"


def test_only_official_https_utas_links_are_clickable():
    assert official_source_url("https://www.utas.edu.au/research")
    assert official_source_url("http://www.utas.edu.au/research") is None
    assert official_source_url("https://example.org/") is None
    assert source_display_title({"project_id": "7", "title": "Test project"}) == "Project 7 — Test project"
    assert official_supervisor_url("https://discover.utas.edu.au/api/users/Soonja.Yeom")
    assert official_supervisor_url("https://example.org/profile") is None


def test_project_card_preserves_supervisor_profile_metadata():
    cards = project_cards_from_sources([{
        "citation_id": "S1", "project_id": "12103", "title": "Example",
        "source_url": "https://www.utas.edu.au/research/degrees/available-projects?id=12103",
        "supervisor_profile": {"canonical_name": "Doctor Soonja Yeom", "school": "School of ICT",
                                "research_fields": ["Cybersecurity"],
                                "discovery_url": "https://discover.utas.edu.au/api/users/Soonja.Yeom"},
    }])
    assert cards[0]["supervisor_profile"]["canonical_name"] == "Doctor Soonja Yeom"
