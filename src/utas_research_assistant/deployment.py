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
        if not manifest_path.is_file():
            raise RuntimeError("Public runtime bundle requires deployment_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("source_scope") != "public_utas_only"
                or manifest.get("private_local_documents_included") is not False):
            raise RuntimeError("Runtime manifest is not public-source-only")
        allowed = {*REQUIRED_FILES, "deployment_manifest.json", "graph_manifest.json", "semantic_passages.json"}
        if any(path.name not in allowed or not path.is_file() or path.is_symlink()
               for path in directory.iterdir()):
            raise RuntimeError("Public runtime bundle contains unexpected artifacts")


def _public_runtime_directory() -> Path:
    """Find a complete public bundle in the source checkout, not site-packages.

    Editable installs normally resolve back to src/. A non-editable install can
    instead rely on the checkout working directory (or one of its ancestors).
    Every candidate must pass the same public-only validation before use.
    """
    package_file = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    candidates = dict.fromkeys([
        package_file.parents[2] / "deployment_data",
        cwd / "deployment_data",
        *(parent / "deployment_data" for parent in cwd.parents),
    ])
    failures = []
    for candidate in candidates:
        try:
            validate_runtime_data(candidate, public_only=True)
        except (RuntimeError, OSError, ValueError) as error:
            # Parse/IO exceptions may include file contents; expose only a safe
            # category, while retaining our own content-free validation errors.
            reason = str(error) if isinstance(error, RuntimeError) else "unreadable runtime manifest or files"
            failures.append(f"{candidate}: {reason}")
        else:
            return candidate.resolve()
    raise RuntimeError("No valid public deployment_data directory found. Checked:\n"
                       + "\n".join(failures))


def resolve_runtime_data(local_dir: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    mode = deployment_mode()
    if mode not in {"local", "public"}:
        raise RuntimeError("UTAS_DEPLOYMENT_MODE must be local or public")
    # Public mode never honors local paths or legacy remote-data settings.
    if mode == "public":
        return _public_runtime_directory()
    candidate = Path(local_dir or root / "data/processed")
    validate_runtime_data(candidate)
    return candidate


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
