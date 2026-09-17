import io
import json
from pathlib import Path
import tarfile

import pytest

from utas_research_assistant.deployment import (
    bundle_manifest, fetch_private_bundle, resolve_runtime_data, validate_runtime_data,
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


def test_public_bundle_rejects_local_artifacts(tmp_path):
    bundle = tmp_path / "bundle"
    make_bundle(bundle)
    (bundle / "local_chunks.json").write_text("[]")
    with pytest.raises(RuntimeError, match="local-document"):
        validate_runtime_data(bundle, public_only=True)


def test_local_mode_resolution_uses_explicit_directory(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    make_bundle(bundle)
    monkeypatch.setenv("UTAS_DEPLOYMENT_MODE", "public")
    monkeypatch.setenv("UTAS_DATA_DIR", str(bundle))
    assert resolve_runtime_data() == bundle


def test_private_repo_loader_requires_token(monkeypatch, tmp_path):
    monkeypatch.setenv("UTAS_DATA_REPO", "owner/data")
    monkeypatch.setenv("UTAS_DATA_REF", "main")
    monkeypatch.delenv("UTAS_DATA_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="UTAS_DATA_TOKEN"):
        fetch_private_bundle(tmp_path)


def test_private_repo_loader_downloads_and_validates_archive(monkeypatch, tmp_path):
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        for name in REQUIRED:
            content = b"[]" if name.endswith(".json") else b"public"
            info = tarfile.TarInfo(f"owner-data-main/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        manifest = json.dumps({"source_scope": "public_utas_only", "private_local_documents_included": False}).encode()
        info = tarfile.TarInfo("owner-data-main/deployment_manifest.json")
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))

    class Response:
        content = archive_bytes.getvalue()
        def raise_for_status(self):
            return None

    monkeypatch.setenv("UTAS_DATA_REPO", "owner/data")
    monkeypatch.setenv("UTAS_DATA_REF", "main")
    monkeypatch.setenv("UTAS_DATA_TOKEN", "test-token")
    calls = []
    def get(*args, **kwargs):
        calls.append(args[0])
        return Response()
    monkeypatch.setattr("utas_research_assistant.deployment.requests.get", get)
    result = fetch_private_bundle(tmp_path)
    assert (result / "deployment_manifest.json").exists()
    assert fetch_private_bundle(tmp_path) == result
    assert len(calls) == 1
