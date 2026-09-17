"""Small single-user SQLite persistence layer for local chat history."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import re


DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "local" / "chat_history.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def conversation_title(question: str, max_length: int = 56) -> str:
    """Create a short deterministic title from the first user question."""
    text = " ".join(str(question).split()).strip(" .?!")
    text = re.sub(r"^(please\s+)?(?:can you|could you|would you)\s+", "", text, flags=re.I)
    if len(text) <= max_length:
        return text or "New chat"
    clipped = text[: max_length + 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return (clipped or text[:max_length]).rstrip() + "…"


def _connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC, id DESC);
        """
    )
    return connection


def init_db(path: Path | str = DEFAULT_DB_PATH) -> None:
    connection = _connect(path)
    connection.close()


def create_conversation(title: str | None = None, *, first_question: str | None = None,
                        path: Path | str = DEFAULT_DB_PATH) -> dict:
    now = _now()
    value = title or (conversation_title(first_question) if first_question else "New chat")
    with _connect(path) as connection:
        cursor = connection.execute(
            "INSERT INTO conversations(title, created_at, updated_at) VALUES (?, ?, ?)",
            (value, now, now),
        )
        row = connection.execute("SELECT * FROM conversations WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)


def list_conversations(path: Path | str = DEFAULT_DB_PATH) -> list[dict]:
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC, id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: int, path: Path | str = DEFAULT_DB_PATH) -> dict | None:
    with _connect(path) as connection:
        conversation = connection.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if conversation is None:
            return None
        messages = connection.execute(
            "SELECT id, role, content, created_at, metadata_json FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,)
        ).fetchall()
    result = dict(conversation)
    result["messages"] = []
    for row in messages:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        except (TypeError, ValueError):
            item["metadata"] = {}
            item.pop("metadata_json", None)
        result["messages"].append(item)
    return result


def add_message(conversation_id: int, role: str, content: str, metadata: dict | None = None,
                *, path: Path | str = DEFAULT_DB_PATH) -> dict:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")
    if not str(content).strip():
        raise ValueError("content must not be blank")
    now = _now()
    encoded = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
    with _connect(path) as connection:
        exists = connection.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if exists is None:
            raise ValueError(f"Conversation {conversation_id} does not exist")
        cursor = connection.execute(
            "INSERT INTO messages(conversation_id, role, content, created_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?)", (conversation_id, role, str(content), now, encoded),
        )
        connection.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
        row = connection.execute("SELECT id, role, content, created_at, metadata_json FROM messages WHERE id = ?",
                                 (cursor.lastrowid,)).fetchone()
    item = dict(row)
    item["metadata"] = json.loads(item.pop("metadata_json"))
    return item


def rename_conversation(conversation_id: int, title: str, *, path: Path | str = DEFAULT_DB_PATH) -> dict | None:
    title = " ".join(str(title).split()).strip() or "New chat"
    with _connect(path) as connection:
        connection.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                           (title, _now(), conversation_id))
        row = connection.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    return dict(row) if row else None


def delete_conversation(conversation_id: int, *, path: Path | str = DEFAULT_DB_PATH) -> bool:
    with _connect(path) as connection:
        cursor = connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    return cursor.rowcount > 0


def clear_all_history(path: Path | str = DEFAULT_DB_PATH) -> None:
    with _connect(path) as connection:
        connection.execute("DELETE FROM conversations")
