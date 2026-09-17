"""Runtime data resolution for local and public deployment modes.

The public mode accepts only a versioned public-UTAS runtime bundle. Local mode
keeps the existing full processed directory, including optional private data.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

import requests


REQUIRED_FILES = ("general_chunks.json", "project_documents.json",
                  "supervisor_documents.json", "utas_research_graph.ttl",
                  "semantic_index.json", "semantic_embeddings.npy")


def _setting(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name)
        return str(value) if value else default
    except Exception:
        return default


def deployment_mode() -> str:
    return (_setting("UTAS_DEPLOYMENT_MODE", "local") or "local").casefold()


def validate_runtime_data(directory: Path, *, public_only: bool = False) -> None:
    directory = Path(directory)
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"Runtime data is incomplete; missing: {', '.join(missing)}")
    if public_only:
        if (directory / "local_chunks.json").exists() or (directory / "local_documents.json").exists():
            raise RuntimeError("Public runtime bundle contains local-document artifacts")
        manifest_path = directory / "deployment_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source_scope") != "public_utas_only" or manifest.get("private_local_documents_included"):
                raise RuntimeError("Runtime manifest is not public-source-only")


def _safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        root = target.resolve()
        for member in tar.getmembers():
            destination = (target / member.name).resolve()
            if root not in destination.parents and destination != root:
                raise RuntimeError("Unsafe path in deployment archive")
        try:
            tar.extractall(target, filter="data")
        except TypeError:  # Python versions before the extraction filter API
            tar.extractall(target)
    candidates = [path for path in target.iterdir() if path.is_dir()]
    if len(candidates) == 1 and (candidates[0] / "deployment_manifest.json").exists():
        for child in candidates[0].iterdir():
            child.replace(target / child.name)
        candidates[0].rmdir()


def fetch_private_bundle(cache_dir: Path | None = None) -> Path:
    repo = _setting("UTAS_DATA_REPO")
    ref = _setting("UTAS_DATA_REF", "main")
    token = _setting("UTAS_DATA_TOKEN")
    if not repo:
        raise RuntimeError("Public deployment needs UTAS_DATA_REPO or UTAS_DATA_DIR")
    if not token:
        raise RuntimeError("UTAS_DATA_TOKEN is required to read the private runtime-data repository")
    cache = Path(cache_dir or os.environ.get("UTAS_DATA_CACHE", Path(tempfile.gettempdir()) / "utas-runtime-data"))
    target = cache / str(ref)
    try:
        validate_runtime_data(target, public_only=True)
        return target
    except RuntimeError:
        pass
    response = requests.get(f"https://api.github.com/repos/{repo}/tarball/{ref}",
                            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                            timeout=30)
    response.raise_for_status()
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", dir=cache, delete=False) as handle:
        handle.write(response.content)
        archive = Path(handle.name)
    try:
        if target.exists():
            import shutil
            shutil.rmtree(target)
        _safe_extract(archive, target)
        validate_runtime_data(target, public_only=True)
        return target
    finally:
        archive.unlink(missing_ok=True)


def resolve_runtime_data(local_dir: Path | None = None) -> Path:
    configured = Path(_setting("UTAS_DATA_DIR")) if _setting("UTAS_DATA_DIR") else None
    candidate = configured or local_dir
    public = deployment_mode() == "public"
    if candidate is None and not public:
        candidate = Path(__file__).resolve().parents[2] / "data" / "processed"
    if candidate is not None:
        try:
            validate_runtime_data(candidate, public_only=public)
            return candidate
        except RuntimeError:
            if configured or public:
                raise
    return fetch_private_bundle()


def bundle_manifest(directory: Path) -> dict:
    files = []
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.name != "deployment_manifest.json":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest})
    return {"bundle_version": datetime.now(timezone.utc).strftime("%Y%m%d"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_scope": "public_utas_only", "private_local_documents_included": False,
            "files": files}
