# Single-repository deployment

Deploy `Hosna-Ara/utas-research-degree-assistant` with `app.py` as the entry point.
The code and reviewed public UTAS runtime bundle are in this one repository.
Set only this application setting in Streamlit Community Cloud secrets:

```toml
UTAS_DEPLOYMENT_MODE = "public"
```

Public mode always uses `deployment_data/` and validates it with
`public_only=True`. It requires a public-only deployment manifest and rejects
local document artifacts, unexpected files, and symlinks. There is no private
runtime-data repository, GitHub token, or remote bundle download. Legacy
`UTAS_DATA_*` settings are unused and can be removed. Public chat history stays
in session state; it does not read or write the local SQLite database.

Local mode is the default (or set `UTAS_DEPLOYMENT_MODE=local`). It uses
`data/processed/`, including optional private/local documents, and permits local
SQLite chat history. Explicit data-directory arguments remain available for local
tools and tests. They cannot override the public runtime directory.

## Updating the runtime bundle

Build public-only assets with `python scripts/build_deployment_bundle.py`.
Review `dist/utas-public-runtime-data/` and its manifest, then copy only these
reviewed files into `deployment_data/`:

- `deployment_manifest.json`
- `general_chunks.json`
- `graph_manifest.json`
- `project_documents.json`
- `semantic_embeddings.npy`
- `semantic_index.json`
- `semantic_passages.json`
- `supervisor_documents.json`
- `utas_research_graph.ttl`

Never copy the full local processed directory or its semantic index. Keep private
assignment documents, their derivatives and evaluations, SQLite history, actual
secrets, and `dist/` excluded from Git. Run the full tests and inspect staged files
before committing an updated public bundle.
