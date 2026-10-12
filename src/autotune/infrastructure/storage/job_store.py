"""Training, benchmark, and pod records, kept in SQLite so they outlive the server."""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

INTERRUPTED_REASON = "The server restarted while this run was in progress, so it stopped."


class JobStore:
    """One JSON record per (kind, id). Lists come back newest first by creation."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS records ("
                " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
                " kind TEXT NOT NULL, id TEXT NOT NULL, record TEXT NOT NULL,"
                " UNIQUE (kind, id))"
            )

    def save(self, kind: str, record_id: str, record: dict[str, Any]) -> None:
        payload = json.dumps(record)
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO records (kind, id, record) VALUES (?, ?, ?)"
                " ON CONFLICT (kind, id) DO UPDATE SET record = excluded.record",
                (kind, record_id, payload),
            )

    def get(self, kind: str, record_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT record FROM records WHERE kind = ? AND id = ?", (kind, record_id)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, kind: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT record FROM records WHERE kind = ? ORDER BY seq DESC", (kind,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def mark_interrupted(self, kind: str) -> None:
        """Runs still marked running were cut off by a restart. Say so instead of showing them live."""
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT id, record FROM records WHERE kind = ?", (kind,)).fetchall()
            for record_id, payload in rows:
                record = json.loads(payload)
                if record.get("status") != "running":
                    continue
                record.update(status="interrupted", error=INTERRUPTED_REASON)
                record.setdefault("logs", []).append(INTERRUPTED_REASON)
                db.execute(
                    "UPDATE records SET record = ? WHERE kind = ? AND id = ?",
                    (json.dumps(record), kind, record_id),
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits on success and is always closed."""
        with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
            yield db
