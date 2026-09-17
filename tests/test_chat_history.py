from pathlib import Path

from utas_research_assistant.chat_history import (
    add_message, conversation_title, create_conversation, delete_conversation,
    get_conversation, init_db, list_conversations, rename_conversation,
)


def test_title_generation_is_deterministic_and_clean():
    assert conversation_title("  Find AI and machine learning PhD projects? ") == "Find AI and machine learning PhD projects"
    assert conversation_title("Please can you explain " + "research " * 20).endswith("…")


def test_sqlite_history_round_trip_and_ordering(tmp_path: Path):
    db = tmp_path / "history.db"
    init_db(db)
    conversation = create_conversation(first_question="Find AI projects", path=db)
    cid = conversation["id"]
    add_message(cid, "user", "Find AI projects", {"scope": "projects"}, path=db)
    add_message(cid, "assistant", "Here are potential matches.", {"project_ids": ["12024"]}, path=db)
    loaded = get_conversation(cid, db)
    assert loaded["title"] == "Find AI projects"
    assert [item["role"] for item in loaded["messages"]] == ["user", "assistant"]
    assert loaded["messages"][1]["metadata"] == {"project_ids": ["12024"]}
    assert list_conversations(db)[0]["id"] == cid


def test_rename_delete_and_reopen_persistence(tmp_path: Path):
    db = tmp_path / "history.db"
    conversation = create_conversation(title="Original", path=db)
    cid = conversation["id"]
    rename_conversation(cid, "Renamed title", path=db)
    assert get_conversation(cid, db)["title"] == "Renamed title"
    assert delete_conversation(cid, path=db)
    assert get_conversation(cid, db) is None
    assert list_conversations(db) == []
