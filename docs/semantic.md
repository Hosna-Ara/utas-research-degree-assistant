# Local semantic baseline

Install with `pip install -e .`, then run:

```bash
python scripts/build_embeddings.py
python scripts/search_semantic.py "English language requirements" --top-k 5
python scripts/probe_semantic.py
python scripts/compare_retrieval.py
```

The build uses the cached `sentence-transformers/all-MiniLM-L6-v2` model when
available, downloading on first use. It encodes passages derived from the BM25
corpus on CPU in batches of 32. Change the
model with `--model NAME` or `UTAS_EMBEDDING_MODEL`, and rebuild. Searches load the
model named in the saved index using local files only. No document vectors are
regenerated during searches. Model weights live in the Hugging Face local cache.

Embeddings are float32 unit vectors; dot products are cosine similarities.
`semantic_embeddings.npy` and `semantic_index.json` are saved in `data/processed/`.
The index contains ordered texts, tokens, metadata, provenance, model details,
build time, and integrity hashes. Searches reject mismatched, reordered, or stale
corpus/vector caches rather than silently mapping results to the wrong documents.
Build time includes loading/downloading the model and encoding, but excludes
writing the two cache files. General chunk deduplication remains unchanged.

General embedding passages target 180–220 words, at most 240 words, with 40-word
overlap. Paragraph boundaries are preferred. MiniLM's 256-token limit (including
special tokens) takes precedence, so some passages are shorter. Every passage is
checked with the actual model tokenizer before encoding; no passages are silently
truncated. Projects remain one embedding each; an oversized project causes an
explicit build error. `semantic_passages.json` stores general passages with their
parent IDs, word offsets, metadata, and full provenance.

The 55 original display chunks are unchanged. Their five duplicate texts still
merge to 50 corpus items with all provenance retained. Passages are built for
these unique parents. The index stores both parent corpus records and ordered
embedding items. Search uses the maximum passage score per parent, returns each
parent once, and displays its original full text. `matched_passage` identifies
the matching span. This changes semantic preparation only; it does not combine
BM25 and semantic scores. Index format version 2 requires rebuilding older caches.

Results are sorted by descending cosine similarity with corpus order breaking
ties. Scores are not calibrated probabilities and have no acceptance threshold.
Empty/punctuation-only queries and nonpositive top-k return no results. Oversized
top-k returns the full corpus. A valid nonempty index is required for search.

Probes import the same question list as BM25. The comparison script measures
overlap by corpus item identity, not repeated document titles, and prints both
rankings without assigning relevance labels or declaring a winner. Tests use
mocked encoders and never download a model.
