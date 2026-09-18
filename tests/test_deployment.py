import json
from pathlib import Path

import pytest

from utas_research_assistant.deployment import (
    bundle_manifest, resolve_runtime_data, validate_runtime_data,
)


REQUIRED = ("general_chunks.json", "project_documents.json", "supervisor_documents.json",
            "utas_research_graph.ttl", "semantic_index.json", "semantic_embeddings.npy")


def make_bundle(path: Path, *, public=True):
    path.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        (path / name).write_bytes(b"[]" if name.endswith(".json") else b"public")
    (path / "deployment_manifest.json").write_text(json.dumps({
        "source_scope": "public_utas_only" if public else "local_full",
        "private_local_documents_included": not public,
    }))


def test_public_bundle_validation_and_manifest(tmp_path):
    bundle = tmp_path / "bundle"
    make_bundle(bundle)
    validate_runtime_data(bundle, public_only=True)
    manifest = bundle_manifest(bundle)
    assert manifest["source_scope"] == "public_utas_only"
    assert manifest["private_local_documents_included"] is False
    assert all("local" not in item["name"] for item in manifest["files"])


@pytest.mark.parametrize("filename", ["local_chunks.json", "local_documents.json"])
def test_public_bundle_rejects_local_artifacts(tmp_path, filename):
    bundle = tmp_path / "bundle"
    make_bundle(bundle)
    (bundle / filename).write_text("[]")
    with pytest.raises(RuntimeError, match="local-document"):
        validate_runtime_data(bundle, public_only=True)


ROOT = Path(__file__).resolve().parents[1]


def test_local_mode_uses_processed(monkeypatch, tmp_path):
    monkeypatch.setenv("UTAS_DEPLOYMENT_MODE", "local")
    assert resolve_runtime_data() == ROOT / "data/processed"
    make_bundle(tmp_path)
    assert resolve_runtime_data(tmp_path) == tmp_path


def test_public_mode_uses_bundled_data(monkeypatch):
    monkeypatch.setenv("UTAS_DEPLOYMENT_MODE", "public")
    monkeypatch.setenv("UTAS_DATA_DIR", str(ROOT / "data/processed"))
    assert resolve_runtime_data(ROOT / "data/processed") == ROOT / "deployment_data"


@pytest.mark.parametrize("manifest", [None, {}, {"source_scope": "public_utas_only"},
    {"source_scope": "local_full", "private_local_documents_included": True}])
def test_public_manifest_required(tmp_path, manifest):
    make_bundle(tmp_path)
    path = tmp_path / "deployment_manifest.json"
    if manifest is None:
        path.unlink()
    else:
        path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        validate_runtime_data(tmp_path, public_only=True)


@pytest.mark.parametrize("filename", ["chat_history.db", "private_profile.txt", "local_manifest.json"])
def test_public_bundle_rejects_unexpected_files(tmp_path, filename):
    make_bundle(tmp_path)
    (tmp_path / filename).write_text("private")
    with pytest.raises(RuntimeError, match="unexpected"):
        validate_runtime_data(tmp_path, public_only=True)


def test_checked_in_bundle_is_public_and_matches_manifest():
    import hashlib
    import numpy as np
    from utas_research_assistant.retrieval.corpus import load_corpus

    directory = ROOT / "deployment_data"
    validate_runtime_data(directory, public_only=True)
    manifest = json.loads((directory / "deployment_manifest.json").read_text())
    assert manifest["projects"] == 219
    assert manifest["supervisors"] == 174
    assert manifest["semantic_vectors"] == 1040
    assert np.load(directory / "semantic_embeddings.npy").shape[0] == 1040
    assert {p.name for p in directory.iterdir()} == {
        "deployment_manifest.json", *(row["name"] for row in manifest["files"])}
    for row in manifest["files"]:
        content = (directory / row["name"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == row["sha256"]
        if not row["name"].endswith(".npy"):
            assert b"LOCAL-" not in content
            assert b"local_document" not in content
            assert b"Applicant B" not in content
    assert all(item.item_type != "local_document" for item in load_corpus(directory).items)


@pytest.fixture
def public_service(monkeypatch):
    from utas_research_assistant.query.planner import OllamaReasoningPlanner
    from utas_research_assistant.service import create_service
    monkeypatch.setenv("UTAS_DEPLOYMENT_MODE", "public")
    monkeypatch.setattr(OllamaReasoningPlanner, "is_available", lambda self: False)
    return create_service()


@pytest.mark.parametrize("question", [
    "What type of applicant is Applicant B?",
    "Using my private applicant profile, summarise my background and main research interests.",
    "What does my private project decision journal say?",
])
def test_private_documents_unavailable_in_public_mode(public_service, question):
    result = public_service.answer_with_evidence(question)
    assert result.response.insufficient_evidence
    assert all(row["item_type"] != "local_document"
               for row in result.evidence.get("ranked_retrieval_evidence", []))
    assert "LOCAL-" not in result.response.model_dump_json()


@pytest.mark.parametrize("question, expected_type", [
    ("Find AI and machine learning PhD projects", "research_project"),
    ("Tell me about Soonja Yeom", "supervisor_profile"),
    ("What English score do I need for a PhD?", "general_chunk"),
])
def test_public_utas_retrieval(public_service, question, expected_type):
    result = public_service.answer_with_evidence(question)
    assert not result.response.insufficient_evidence
    assert any(row["item_type"] == expected_type
               for row in result.evidence.get("ranked_retrieval_evidence", []))


@pytest.mark.parametrize('installed_package', [False, True])
@pytest.mark.parametrize('working_directory', ['checkout', 'nested', 'unrelated'])
def test_cloud_checkout_resolution(monkeypatch, tmp_path, installed_package, working_directory):
    from utas_research_assistant import deployment

    checkout = tmp_path / 'mount/src/utas-research-degree-assistant'
    bundle = checkout / 'deployment_data'
    make_bundle(bundle)
    package_file = (tmp_path / 'venv/lib/python3.12/site-packages/utas_research_assistant/deployment.py'
                    if installed_package else checkout / 'src/utas_research_assistant/deployment.py')
    monkeypatch.setattr(deployment, '__file__', str(package_file))
    cwd = {'checkout': checkout, 'nested': checkout / 'app/work', 'unrelated': tmp_path / 'other'}[working_directory]
    cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(cwd)
    monkeypatch.setenv('UTAS_DEPLOYMENT_MODE', 'public')
    if installed_package and working_directory == 'unrelated':
        # No checkout anchor is available; fail clearly rather than guess.
        with pytest.raises(RuntimeError, match='No valid public deployment_data directory found'):
            resolve_runtime_data()
    else:
        assert resolve_runtime_data() == bundle


def test_public_resolution_skips_incomplete_package_candidate(monkeypatch, tmp_path):
    from utas_research_assistant import deployment

    package_root = tmp_path / 'installed'
    incomplete = package_root / 'deployment_data'
    incomplete.mkdir(parents=True)
    (incomplete / 'general_chunks.json').write_text('[]')
    checkout = tmp_path / 'mount/src/utas-research-degree-assistant'
    make_bundle(checkout / 'deployment_data')
    monkeypatch.setattr(deployment, '__file__', str(package_root / 'src/utas_research_assistant/deployment.py'))
    monkeypatch.chdir(checkout)
    monkeypatch.setenv('UTAS_DEPLOYMENT_MODE', 'public')
    assert resolve_runtime_data() == checkout / 'deployment_data'


def test_public_resolution_rejects_private_bundle(monkeypatch, tmp_path):
    from utas_research_assistant import deployment

    checkout = tmp_path / 'mount/src/utas-research-degree-assistant'
    make_bundle(checkout / 'deployment_data')
    (checkout / 'deployment_data/local_chunks.json').write_text('[]')
    monkeypatch.setattr(deployment, '__file__', str(checkout / 'src/utas_research_assistant/deployment.py'))
    monkeypatch.chdir(checkout)
    monkeypatch.setenv('UTAS_DEPLOYMENT_MODE', 'public')
    with pytest.raises(RuntimeError, match='local-document artifacts'):
        resolve_runtime_data()


def test_cloud_public_service_and_safe_diagnostics(monkeypatch, tmp_path, caplog):
    import logging
    import shutil
    from collections import Counter
    from utas_research_assistant import deployment, service
    from utas_research_assistant.query.planner import OllamaReasoningPlanner

    checkout = tmp_path / 'mount/src/utas-research-degree-assistant'
    bundle = checkout / 'deployment_data'
    shutil.copytree(ROOT / 'deployment_data', bundle)
    monkeypatch.setattr(deployment, '__file__', str(checkout / 'src/utas_research_assistant/deployment.py'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('UTAS_DEPLOYMENT_MODE', 'public')
    monkeypatch.setenv('UTAS_DATA_TOKEN', 'do-not-log-this-secret')
    monkeypatch.setattr(OllamaReasoningPlanner, 'is_available', lambda self: False)
    with caplog.at_level(logging.INFO, logger=service.__name__):
        public = service.create_service()
    assert public.processed_dir == bundle
    counts = Counter(item.item_type for item in public.router.retriever.corpus.items)
    assert counts['general_chunk'] > 0
    assert counts['research_project'] == 219
    assert counts['supervisor_profile'] > 0
    assert counts['local_document'] == 0
    assert len(public.router.graph) > 0
    assert public.router.retriever.semantic.vectors.shape[0] == 1040
    result = public.answer_with_evidence('What English score do I need for a PhD?')
    assert not result.response.insufficient_evidence
    messages = [record.getMessage() for record in caplog.records if record.name == service.__name__]
    assert messages == ['Deployment mode: public', f'Resolved runtime data directory: {bundle}',
                        *(f'Runtime file {name} exists: True' for name in REQUIRED)]
    assert 'do-not-log-this-secret' not in caplog.text
