# Session-only CV privacy audit

Audit scope: the final personalised-matching changes, using synthetic CVs only.

- Uploads are read from Streamlit's in-memory `UploadedFile`; PDF extraction uses
  a byte stream and DOCX extraction uses `BytesIO`/ZIP/XML. There are no temporary
  document files, corpus writes, generated embeddings for persistence, or caches
  of applicant data. File size, page count, expanded ZIP size and text length are
  bounded. External XML entities and encrypted/unreadable PDFs are rejected.
- Applicant facts, ranked entities and personalised answers belong
  to the browser session. Saving/replacing/removing a CV resets derived chat
  context. Selecting a file alone does not extract or activate a profile. Removing it also rotates the uploader widget identity. Streamlit may
  retain a disconnected session briefly for reconnection before memory expiry.
- The session privacy flag disables SQLite conversation creation and message
  writes before any CV-derived turn. It stays enabled after CV removal and new
  chats in that session. Normal local history is unchanged in fresh sessions;
  public mode never accesses local history.
- Qwen extraction is optional. Its endpoint is fixed to loopback; environment
  proxies and HTTP redirects are disabled. Public extraction never invokes it.
  All other CV-session answers use the existing deterministic planner, graph,
  hybrid retriever and grounded answer fallback, with per-call diagnostic objects.
- The sidebar displays only saved-profile status, not extracted applicant text.
  Filenames use plain text, preventing Markdown image requests to third-party URLs.
- Cached services contain only corpus/graph/model assets. No applicant profile
  is assigned to a cached service. Query encoding is in-process; BM25 remains
  available without the semantic model. Exceptions log only their type, not
  applicant text, prompts, contact information or traceback content.
- Public `deployment_data` files still match every checked-in manifest checksum.
  No private local chunks, CV documents, SQLite database or unexpected artifacts
  are present in that bundle. Real service-construction tests confirm zero private
  documents in PUBLIC and ten private document identities in LOCAL.
- A synthetic session sentinel was absent from `data/raw`, `data/processed`,
  `deployment_data` and `data/local`. Git tracks no PDF/DOCX uploads. Synthetic test
  outputs/browser artifacts are ignored; no real CV was used in tests.
- Automated checks cover in-memory PDF/DOCX extraction, invalid uploads,
  deterministic/Qwen paths, no network access in public CV matching, no file
  writes in the CV pipeline, no profile in shared service state, SQLite exclusion,
  rerun/click flow, replacement, removal, and fresh-session isolation.

Validation limits: Streamlit/AppTest exercised the full interaction sequence.
LOCAL and PUBLIC Streamlit server health checks returned HTTP 200. A real Chromium
visual inspection could not run because the sandbox denied browser process setup;
manual desktop/mobile inspection remains recommended. This audit does not claim
secure erasure from operating-system memory or an audit of an independently
configured Ollama server's own diagnostics.
