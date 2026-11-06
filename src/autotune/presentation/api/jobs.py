"""Background jobs: run work on a thread, keep the live record safe, and persist every change."""
from __future__ import annotations

import copy
import logging
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from autotune.infrastructure.storage.job_store import JobStore
from autotune.presentation.errors import EXPECTED_FAILURES

logger = logging.getLogger(__name__)


class JobHandle:
    """What a unit of work sees: update fields and append log lines on its own job."""

    def __init__(self, runner: JobRunner, job_id: str) -> None:
        self._runner = runner
        self.job_id = job_id

    def update(self, **fields: Any) -> None:
        self._runner._update(self.job_id, fields)

    def log(self, line: str) -> None:
        self._runner._log(self.job_id, line)


class JobRunner:
    """Runs jobs of one kind. A finished job leaves the live map after its final save,
    and `snapshot` then reads it from the store."""

    def __init__(self, store: JobStore, kind: str, *, id_prefix: str) -> None:
        self._store = store
        self._kind = kind
        self._id_prefix = id_prefix
        self._lock = threading.Lock()
        self._live: dict[str, dict[str, Any]] = {}
        self._threads: dict[str, threading.Thread] = {}

    def start(self, fields: dict[str, Any], work: Callable[[JobHandle], dict[str, Any] | None]) -> str:
        job_id = f"{self._id_prefix}-{uuid.uuid4().hex[:12]}"
        record = {**fields, "id": job_id, "status": "running", "created_at": _now(), "logs": []}
        with self._lock:
            self._live[job_id] = record
            self._save_locked(job_id)
        thread = threading.Thread(target=self._run, args=(job_id, work), daemon=True)
        self._threads[job_id] = thread
        thread.start()
        return job_id

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._live.get(job_id)
            if record is not None:
                return copy.deepcopy(record)
        return self._store.get(self._kind, job_id)

    def live(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._live

    def join(self, job_id: str, timeout: float | None = None) -> None:
        """Wait for a job's thread to end. Tests use it to avoid sleeping."""
        thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout)

    def _run(self, job_id: str, work: Callable[[JobHandle], dict[str, Any] | None]) -> None:
        outcome: dict[str, Any]
        try:
            result = work(JobHandle(self, job_id))
            outcome = {"status": "completed", **(result or {})}
        except EXPECTED_FAILURES as exc:
            logger.warning("%s job %s failed: %s", self._kind, job_id, exc)
            outcome = {"status": "failed", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - a worker thread must never die with the job stuck on running
            logger.exception("%s job %s crashed", self._kind, job_id)
            outcome = {"status": "failed", "error": str(exc)}
        with self._lock:
            self._live[job_id].update(outcome)
            self._save_locked(job_id)
            del self._live[job_id]

    def _update(self, job_id: str, fields: dict[str, Any]) -> None:
        with self._lock:
            self._live[job_id].update(fields)
            self._save_locked(job_id)

    def _log(self, job_id: str, line: str) -> None:
        with self._lock:
            self._live[job_id]["logs"].append(line)
            self._save_locked(job_id)

    def _save_locked(self, job_id: str) -> None:
        # Saved under the lock so concurrent writers cannot persist an older state over a newer one.
        self._store.save(self._kind, job_id, copy.deepcopy(self._live[job_id]))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
