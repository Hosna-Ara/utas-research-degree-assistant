import json
from datetime import datetime, timezone

from utas_research_assistant.ingestion.supervisor_profiles import (
    SupervisorTarget,
    candidate_profile_ids,
    clean_bio_html,
    normalize_name,
    normalize_profile,
    profile_matches,
    strip_academic_titles,
    load_cached_profile,
    ict_quality_gate,
)


def target(name="Doctor Soonja Yeom"):
    return SupervisorTarget(
        canonical_project_supervisor_name=name,
        normalized_name=normalize_name(name),
        project_ids=["12103"],
        research_categories=["Information and Communication Technology"],
        is_ict_supervisor=True,
    )


def test_title_stripping_and_exact_name_normalization():
    assert strip_academic_titles("Associate Professor Quan Bai") == "Quan Bai"
    assert normalize_name("Dr. Soonja   Yeom") == "soonja yeom"
    assert normalize_name("Doctor Soonja Yeom") == normalize_name("Soonja Yeom")


def test_candidate_ids_support_multi_part_names():
    candidates = candidate_profile_ids("Professor Kristy de Salas")
    assert "Kristy.De.Salas" in candidates
    assert "Kristy.DeSalas" in candidates


def test_profile_requires_conservative_exact_match():
    assert profile_matches({"firstName": "Soonja", "lastName": "Yeom"}, "Doctor Soonja Yeom")
    assert not profile_matches({"firstName": "Soonja", "lastName": "Young"}, "Doctor Soonja Yeom")


def test_bio_html_is_cleaned_without_rewriting():
    value = clean_bio_html("<p>First paragraph.</p><p>Second <b>paragraph</b>.</p><script>bad()</script>")
    assert value == "First paragraph.\nSecond paragraph."


def test_normalized_profile_preserves_projects_and_external_links():
    payload = {
        "firstName": "Soonja", "lastName": "Yeom", "title": "Doctor",
        "tabSummaryAbout": "<p>Research bio.</p>", "tags": ["AI", "security"],
        "orcid": "0000-0000", "addresses": [{"school": "Engineering", "department": "ICT"}],
        "personalWebsites": ["https://linkedin.com/in/soonja", "https://scholar.google.com/citations?user=x"],
        "positions": ["Lecturer"], "degrees": ["PhD"],
    }
    profile = normalize_profile(payload, target(), "Soonja.Yeom", datetime.now(timezone.utc))
    assert profile.related_project_ids == ["12103"]
    assert profile.is_ict_supervisor is True
    assert profile.linkedin_url.startswith("https://linkedin.com")
    assert profile.google_scholar_url.startswith("https://scholar.google.com")


def test_validated_raw_cache_is_reused(tmp_path):
    payload = {"firstName": "Soonja", "lastName": "Yeom", "tabSummaryAbout": "<p>Bio</p>"}
    path = tmp_path / "Soonja.Yeom.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    cached = load_cached_profile(path, target())
    assert cached is not None
    assert cached.discovery_profile_id == "Soonja.Yeom"


def test_mismatched_cached_profile_is_rejected(tmp_path):
    path = tmp_path / "Soonja.Yeom.json"
    path.write_text(json.dumps({"firstName": "Other", "lastName": "Person"}), encoding="utf-8")
    assert load_cached_profile(path, target()) is None


def test_ict_quality_gate_requires_every_ict_target():
    profile = normalize_profile({"firstName": "Soonja", "lastName": "Yeom"}, target(), "Soonja.Yeom")
    assert ict_quality_gate([target()], [profile])
    other = SupervisorTarget(canonical_project_supervisor_name="Doctor Quan Bai", normalized_name="quan bai", is_ict_supervisor=True)
    assert not ict_quality_gate([target(), other], [profile])
