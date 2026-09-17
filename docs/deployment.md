# Deployment data architecture

The public application repository contains code and tests. Runtime data for a
public deployment is built separately with `scripts/build_deployment_bundle.py`
from the public UTAS project, general-document, supervisor, and graph artifacts.
The resulting bundle is intended for a separate private data repository; this
project does not create or upload that repository.

Set `UTAS_DEPLOYMENT_MODE=public`, `UTAS_DATA_REPO`, `UTAS_DATA_REF`, and
`UTAS_DATA_TOKEN` through deployment secrets. The token is used only for a
read-only GitHub API archive request. `UTAS_DATA_DIR` may point directly to a
validated bundle for local deployment simulation. Data is resolved once when
the cached service starts, rather than per question.

Local development keeps the existing full mode: local `data/processed` data is
used first and may include private assignment documents. Personal SQLite chat
history is local-only. Public mode never loads `local_chunks.json`, private
assignment documents, or Hosna's local chat-history database.

Build a bundle locally with:

```bash
python scripts/build_deployment_bundle.py
```

Review the manifest and privacy checks before placing the resulting
`dist/utas-public-runtime-data/` directory in a separate private runtime-data
repository. Never commit private local documents, local indexes, or secrets.
