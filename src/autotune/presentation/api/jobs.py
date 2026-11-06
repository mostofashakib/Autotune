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

# The run list leaves out what only the detail view needs.
_DETAIL_KEYS = {"logs", "loss_history", "report"}


def summary(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in _DETAIL_KEYS}


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
        record = {**fields, "job_id": job_id, "status": "running", "created_at": _now(), "logs": []}
        with self._lock:
            self._store.save(self._kind, job_id, record)
            self._live[job_id] = record
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
        self._finish(job_id, outcome)

    def _finish(self, job_id: str, outcome: dict[str, Any]) -> None:
        """Persist the final record. A failing save is logged, and the job leaves the live map regardless."""
        try:
            self._commit(job_id, lambda record: record.update(outcome))
        except Exception:  # noqa: BLE001 - nothing may escape the worker thread
            logger.exception("%s job %s: final save failed", self._kind, job_id)
        finally:
            with self._lock:
                self._live.pop(job_id, None)
            self._threads.pop(job_id, None)

    def _update(self, job_id: str, fields: dict[str, Any]) -> None:
        self._commit(job_id, lambda record: record.update(fields))

    def _log(self, job_id: str, line: str) -> None:
        self._commit(job_id, lambda record: record["logs"].append(line))

    def _commit(self, job_id: str, change: Callable[[dict[str, Any]], object]) -> None:
        """Apply a change to a copy, save it, and only then make it the live record.

        Saved under the lock so concurrent writers cannot persist an older state over a newer one.
        A failed save raises and leaves the live record as it was.
        """
        with self._lock:
            updated = copy.deepcopy(self._live[job_id])
            change(updated)
            self._store.save(self._kind, job_id, updated)
            self._live[job_id] = updated


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
