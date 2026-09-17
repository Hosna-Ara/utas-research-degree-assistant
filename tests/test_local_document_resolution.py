import pytest

from utas_research_assistant.retrieval.corpus import CorpusItem
from utas_research_assistant.retrieval.local_documents import resolve_local_document


def local(document_id, title="Example document"):
    return CorpusItem("local_document", "", [],
                      {"document_id": document_id, "title": title}, [])


@pytest.mark.parametrize("reference", [
    "my private applicant profile", "my private profile", "private applicant profile",
    "MY PRIVATE APPLICANT PROFILE", "LOCAL-HOSNA-PROFILE-01", "Applicant research notes",
])
def test_alias_id_and_title_resolution(reference):
    item = local("LOCAL-HOSNA-PROFILE-01", "Applicant research notes")
    assert resolve_local_document(f"Summarise {reference}.", [item, item]) == item.metadata["document_id"]


def test_ambiguous_and_generic_references_fall_back():
    items = [local("LOCAL-HOSNA-PROFILE-01"), local("LOCAL-BUSINESS-PERSONA-02")]
    assert resolve_local_document("Compare my private profile with Applicant B", items) is None
    assert resolve_local_document("What are my research interests?", items) is None
    assert resolve_local_document("Applicant Brown", items) is None


def test_unavailable_private_documents_cannot_be_resolved():
    assert resolve_local_document("my private applicant profile", []) is None
    public = CorpusItem("general_chunk", "", [], {
        "document_id": "public", "title": "my private applicant profile",
    }, [])
    assert resolve_local_document("my private applicant profile", [public]) is None
