"""Resolve explicit local references against eligible corpus metadata only."""

from collections.abc import Iterable

from utas_research_assistant.retrieval.corpus import CorpusItem, tokenize


LOCAL_DOCUMENT_ALIASES = {
    "LOCAL-HOSNA-PROFILE-01": ("private applicant profile", "private profile"),
    "LOCAL-BUSINESS-PERSONA-02": ("Applicant B", "business-focused synthetic persona"),
    "LOCAL-DECISION-JOURNAL-07": ("private project decision journal",),
}


def _mentions(query: str, reference: str) -> bool:
    tokens = tokenize(reference)
    return bool(tokens) and f" {' '.join(tokens)} " in f" {' '.join(tokenize(query))} "


def has_local_document_reference(query: str) -> bool:
    """Routing hint only; retrieval must still resolve against eligible items."""
    return any(_mentions(query, reference)
               for document_id, aliases in LOCAL_DOCUMENT_ALIASES.items()
               for reference in (document_id, *aliases))


def resolve_local_document(query: str, items: Iterable[CorpusItem]) -> str | None:
    """Return a unique referenced ID; absent or ambiguous references fall back.

    Match whole token sequences, accepting punctuation/case variations. Aliases
    augment document titles and IDs without making unavailable documents visible.
    """
    matches = set()
    for item in items:
        if item.item_type != "local_document":
            continue
        document_id = item.metadata.get("document_id")
        if not document_id:
            continue
        references = (document_id, item.metadata.get("title", ""),
                      *LOCAL_DOCUMENT_ALIASES.get(document_id, ()))
        for reference in references:
            if _mentions(query, reference):
                matches.add(document_id)
                break
    return next(iter(matches)) if len(matches) == 1 else None
