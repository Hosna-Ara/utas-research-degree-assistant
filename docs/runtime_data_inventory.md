# Runtime data inventory

## Public deployment inputs

The public bundle contains only public-source artifacts:

- `general_chunks.json` — parsed public UTAS research-degree pages.
- `project_documents.json` — validated public UTAS advertised projects.
- `supervisor_documents.json` — normalized public UTAS Discover profiles.
- `utas_research_graph.ttl` — RDF graph built from public project/profile data.
- `graph_manifest.json` — aggregate graph statistics.
- `semantic_index.json` and `semantic_embeddings.npy` — regenerated from the
  public-only corpus; they must never be copied from a full local index.

The application code, tokenizer, BM25 index construction, planner, graph
loader, and answer-generation code are regeneratable application assets.

## Local-only inputs

Local full mode may additionally load `data/processed/local_chunks.json` and
the ten assignment documents under `data/raw/private/`. Their derived semantic
index and any local evaluation artifacts are private as well. Personal
`data/local` SQLite history is never part of a deployment bundle.

## Not runtime data

Raw acquisition snapshots, audit reports, probe outputs, tests, and source
scripts are not required by the deployed question-answering service. They are
kept outside the public runtime bundle.
