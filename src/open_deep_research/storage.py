"""Local SQLite persistence for research runs and settings."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


DEFAULT_DB_PATH = ".local/open_deep_research.sqlite3"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class ResearchStorage:
    """Small SQLite repository for local-first research data."""

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        path = Path(db_path or os.getenv("OPEN_DEEP_RESEARCH_DB", DEFAULT_DB_PATH))
        if not path.is_absolute():
            path = Path.cwd() / path
        self.db_path = path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create local database tables when they do not exist."""
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'research',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES research_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES research_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    title TEXT,
                    url TEXT NOT NULL,
                    snippet TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES research_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES research_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS api_key_refs (
                    provider TEXT PRIMARY KEY,
                    keyring_service TEXT NOT NULL,
                    keyring_username TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES research_runs(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_run_id ON messages(run_id);
                CREATE INDEX IF NOT EXISTS idx_notes_run_id ON notes(run_id);
                CREATE INDEX IF NOT EXISTS idx_sources_run_id ON sources(run_id);
                CREATE INDEX IF NOT EXISTS idx_run_events_run_id
                    ON run_events(run_id, id);
                """
            )

    def create_run(self, query: str, settings: dict[str, Any]) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO research_runs (
                    id, query, status, settings_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, query, "running", json.dumps(settings), now, now),
            )
            conn.execute(
                """
                INSERT INTO messages (run_id, role, content, mode, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, "user", query, "research", now),
            )
            conn.execute(
                """
                INSERT INTO run_events (
                    run_id, event_type, message, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    "created",
                    "Research run created.",
                    json.dumps({"status": "running"}),
                    now,
                ),
            )
        return self.get_run(run_id)

    def update_run_status(
        self,
        run_id: str,
        status: str,
        error: str | None = None,
        completed: bool = False,
    ) -> None:
        now = utc_now()
        completed_at = now if completed else None
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE research_runs
                SET status = ?, error = ?, updated_at = ?,
                    completed_at = COALESCE(?, completed_at)
                WHERE id = ?
                """,
                (status, error, now, completed_at, run_id),
            )

    def reset_running_runs(self, reason: str) -> int:
        """Mark in-flight runs failed after backend restart or task loss."""
        now = utc_now()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id FROM research_runs WHERE status = ?",
                ("running",),
            ).fetchall()
            cursor = conn.execute(
                """
                UPDATE research_runs
                SET status = ?, error = ?, updated_at = ?, completed_at = ?
                WHERE status = ?
                """,
                ("failed", reason, now, now, "running"),
            )
            conn.executemany(
                """
                INSERT INTO run_events (
                    run_id, event_type, message, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["id"],
                        "failed",
                        reason,
                        json.dumps({"status": "failed"}),
                        now,
                    )
                    for row in rows
                ],
            )
        return cursor.rowcount

    def append_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a run event and return it."""
        now = utc_now()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO run_events (
                    run_id, event_type, message, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, event_type, message, json.dumps(payload or {}), now),
            )
            event_id = cursor.lastrowid
        return {
            "id": event_id,
            "run_id": run_id,
            "event_type": event_type,
            "message": message,
            "payload": payload or {},
            "created_at": now,
        }

    def list_events(
        self,
        run_id: str,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """List run events after an optional event id."""
        with self.connect() as conn:
            if after_id is None:
                rows = conn.execute(
                    """
                    SELECT id, run_id, event_type, message, payload_json, created_at
                    FROM run_events
                    WHERE run_id = ?
                    ORDER BY id ASC
                    """,
                    (run_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, run_id, event_type, message, payload_json, created_at
                    FROM run_events
                    WHERE run_id = ? AND id > ?
                    ORDER BY id ASC
                    """,
                    (run_id, after_id),
                ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["payload"] = json.loads(event.pop("payload_json") or "{}")
            events.append(event)
        return events

    def list_runs(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT r.*, reports.content AS report
                FROM research_runs r
                LEFT JOIN reports ON reports.run_id = r.id
                ORDER BY r.created_at DESC
                """
            ).fetchall()
        return [self._run_row_to_dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT r.*, reports.content AS report
                FROM research_runs r
                LEFT JOIN reports ON reports.run_id = r.id
                WHERE r.id = ?
                """,
                (run_id,),
            ).fetchone()
        return self._run_row_to_dict(row) if row else None

    def get_run_detail(self, run_id: str) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        if not run:
            return None
        with self.connect() as conn:
            messages = conn.execute(
                """
                SELECT id, role, content, mode, created_at
                FROM messages
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
            notes = conn.execute(
                """
                SELECT id, kind, content, created_at
                FROM notes
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
            sources = conn.execute(
                """
                SELECT id, title, url, snippet, created_at
                FROM sources
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        run["messages"] = [dict(row) for row in messages]
        run["notes"] = [dict(row) for row in notes]
        run["sources"] = [dict(row) for row in sources]
        return run

    def append_message(
        self,
        run_id: str,
        role: str,
        content: str,
        mode: str = "research",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (run_id, role, content, mode, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, role, content, mode, utc_now()),
            )

    def save_report(self, run_id: str, content: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO reports (run_id, content, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    content = excluded.content,
                    created_at = excluded.created_at
                """,
                (run_id, content, utc_now()),
            )

    def replace_notes(
        self,
        run_id: str,
        notes: list[str],
        raw_notes: list[str],
        research_brief: str | None,
    ) -> None:
        now = utc_now()
        rows = []
        if research_brief:
            rows.append((run_id, "research_brief", research_brief, now))
        rows.extend((run_id, "note", note, now) for note in notes if note)
        rows.extend((run_id, "raw_note", note, now) for note in raw_notes if note)
        with self.connect() as conn:
            conn.execute("DELETE FROM notes WHERE run_id = ?", (run_id,))
            conn.executemany(
                """
                INSERT INTO notes (run_id, kind, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )

    def replace_sources(self, run_id: str, sources: list[dict[str, str | None]]) -> None:
        now = utc_now()
        seen: set[str] = set()
        rows = []
        for source in sources:
            url = source.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            rows.append(
                (
                    run_id,
                    source.get("title"),
                    url,
                    source.get("snippet"),
                    now,
                )
            )
        with self.connect() as conn:
            conn.execute("DELETE FROM sources WHERE run_id = ?", (run_id,))
            conn.executemany(
                """
                INSERT INTO sources (run_id, title, url, snippet, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_settings(self) -> dict[str, Any]:
        with self.connect() as conn:
            rows = conn.execute("SELECT key, value_json FROM settings").fetchall()
        return {row["key"]: json.loads(row["value_json"]) for row in rows}

    def update_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO settings (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                [(key, json.dumps(value), now) for key, value in settings.items()],
            )
        return self.get_settings()

    def save_api_key_ref(
        self,
        provider: str,
        keyring_service: str,
        keyring_username: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO api_key_refs (
                    provider, keyring_service, keyring_username, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    keyring_service = excluded.keyring_service,
                    keyring_username = excluded.keyring_username,
                    updated_at = excluded.updated_at
                """,
                (provider, keyring_service, keyring_username, utc_now()),
            )

    def list_api_key_refs(self) -> list[dict[str, str]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT provider, keyring_service, keyring_username, updated_at
                FROM api_key_refs
                ORDER BY provider
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def _run_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["settings"] = json.loads(data.pop("settings_json") or "{}")
        return data
