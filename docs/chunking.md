# General document chunks

Run `python scripts/build_chunks.py` after installing the project. The script
reads only `data/processed/general_documents.json` and writes
`general_chunks.json` and `chunk_manifest.json` in the same directory.

Chunks target 400 whitespace-delimited words, normally 250–500, with a
70-word suffix repeated from the preceding chunk. Paragraphs stay intact
where possible. Because saved text has no HTML heading tags, short blocks of
at most 20 words without sentence-ending periods, exclamation marks, or
semicolons are treated as likely headings and kept with following text.
This includes short FAQ questions. Heading detection is a heuristic.

Oversized paragraphs or heading groups use a word-boundary fallback at 430
words to leave room for overlap. Short documents, final chunks, and boundaries
before large paragraphs can produce chunks below 250 words. Overlap may begin
within a paragraph; the new content remains paragraph-aware.

Each input document is processed independently, including individual PDF page
records. Chunk indices start at zero within each document. IDs hash document
ID, chunk index, and chunk text, so identical inputs produce identical IDs
regardless of run time or document order. Changed chunk text changes its ID.
Empty documents are listed in the manifest; invalid input and duplicate
document IDs cause a clear error before outputs are written.

Run `python scripts/build_project_documents.py` to create one separate retrieval
document per structured project in `project_documents.json`. Its text uses
labelled source fields, and its metadata retains the original values, including
description. Missing fields are omitted from the text and remain empty in metadata.
The source `projects.json` is never written.

Retrieval-only `funding_status` is `no_stipend` for explicit “No stipend” wording,
`funded` for a currency-marked amount, and `unknown` otherwise. “No stipend”
takes precedence if both occur. These are text-based retrieval labels, not a
guarantee of funding. Project document IDs use the project ID (or source URL if
missing), so they remain stable when descriptive fields change.
