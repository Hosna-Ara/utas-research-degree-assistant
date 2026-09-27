# UTAS Research Degree Assistant

A Python project for a research-degree discovery chatbot using official University of Tasmania public information.

Current status: local retrieval, graph reasoning, grounded answers, supervisor profiles, and a Streamlit prototype are implemented. Chat history is stored locally in SQLite and is not committed to the repository.

This is an unofficial student-built prototype. Users should verify information with official UTAS sources.

## Personalised CV Matching

Open **Workspace → CV / Profile** in the left sidebar and select a text-based
**PDF or DOCX** (up to 5 MB, PDF up to 40 pages). Click **Save CV** to extract and
activate the session profile; selecting a file alone does not activate it. A compact
status and project, supervisor, gap and best-match actions appear after saving.
The extracted profile is not displayed. Save a replacement CV to update it, or
click **Remove CV** to return to generic mode and clear derived conversation state.
Scanned PDFs require OCR before upload.

**Session-only privacy:** upload bytes, extracted text, profiles and personalised
conversations are never written to the corpus, deployment bundle, SQLite history,
logs or a global cache. Attaching a CV makes subsequent chats in that browser
session ephemeral, including after clearing it; normal local history remains
available in a fresh session. Clear/remove CV also clears its derived conversation
and references. A new browser session starts without the CV/profile. Streamlit may
retain disconnected session memory briefly for reconnection until session expiry.
No CV content is sent to an external API. Optional Qwen extraction uses only the
loopback Ollama endpoint, disables proxies/redirects and rejects unquoted additions.

```text
CV Upload → Text Extraction → Applicant Profile
→ Semantic + Structured Matching → Project/Supervisor Ranking
→ Grounded Explanation → Contextual Next-Step Suggestions
```

Project alignment uses five weights: research/topic **30**, skills/methods **25**,
academic/domain **15**, supervisor research **15**, practical preferences **15**.
For text dimensions, points are weight × the fraction of distinct applicant
keywords found in the corresponding source text (case-insensitive, with common
words removed). Practical points use explicit degree, location, student type and
funding evidence; unverified requested constraints reduce practical coverage.
Missing applicant/source evidence is shown as unknown. Alignment is earned points
divided by supported weight; evidence coverage is supported weight out of 100.
Scores are rounded heuristic indicators, **not admission probabilities or verified
eligibility assessments**. The score breakdown and source excerpts are expandable.
Existing semantic/BM25 reciprocal-rank fusion orders ties; BM25 continues when the
semantic model is unavailable. Supervisor ranking uses the topic, skills and
academic dimensions of profile/related-project evidence, with unavailable
practical suitability marked unknown; no publication metrics are used. Supervisor
project relationships are verified using the existing RDF graph.

Both modes support matching and dynamic, rule-based suggestion buttons without
Ollama. LOCAL retains the existing public and ten private documents, with optional
local Qwen for extraction. PUBLIC uses only `deployment_data/`; no private corpus
is loaded. General follow-ups in a CV session reuse the existing deterministic
planner, graph and answer evidence guards without model network calls. Suggested
questions are stored with each session answer and submitted through the normal
chat callback. Ranked references such as “project 1” remain stable across detail
and supervisor follow-ups.

Limitations: the knowledge base is the September 2026 snapshot, not a live vacancy
feed. Extraction is conservative and may miss unusual CV layouts or terminology;
check recommendation evidence and save a corrected CV if needed. Keyword scoring cannot establish proficiency, qualification
recognition or supervision availability. A topic present in a project but absent
from the CV is an **unknown**, not proof of a skills gap. Verify current funding,
eligibility and deadlines through the cited official sources.

LoRA is not currently used: defensible fine-tuning would require a labelled
applicant–project dataset. Domain-specific LoRA may be explored if a sufficiently
large labelled applicant–project matching dataset becomes available.

Tests: `.venv/bin/python -m pytest -q`. CV tests use generated synthetic documents,
including upload callbacks and Streamlit/AppTest follow-up, rerun and clearing
flows. No real applicant CV belongs in fixtures or deployment artifacts.
