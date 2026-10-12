"""The training API runs the real backend and reports only what the trainer logged."""
from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from autotune.core.domain.models import CheckpointResult, JobStatus, TrainingProgress
from autotune.presentation.api import app as api


class FakeBackend:
    """Stands in for TrlPeftBackend: logs the progress it is given, then returns a checkpoint."""

    def __init__(self, progress=(), error: Exception | None = None, release: threading.Event | None = None):
        self.progress, self.error, self.release = list(progress), error, release
        self.cancelled = []

    def train(self, job, dataset, on_progress=None):
        for update in self.progress:
            on_progress(update)
        if self.release is not None:
            self.release.wait(5)
        if self.error is not None:
            raise self.error
        status = JobStatus.CANCELLED if job.id in self.cancelled else JobStatus.COMPLETED
        return CheckpointResult(job_id=job.id, checkpoint_dir=job.output_dir, final_loss=0.31, total_steps=20, status=status)

    def cancel(self, job_id):
        self.cancelled.append(job_id)


@pytest.fixture
def client():
    return TestClient(api.create_app())


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "sft.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(12)]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _start(client, dataset, tmp_path, **overrides):
    body = {"dataset_name": str(dataset), "base_model": "Qwen/Qwen2.5-0.5B-Instruct", "output_dir": str(tmp_path / "ckpt"), **overrides}
    return client.post("/api/training/start", json=body)


def _wait(client, job_id):
    for _ in range(300):
        job = client.get(f"/api/training/status/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.01)
    raise AssertionError("training job never finished")


def test_a_training_job_reports_the_progress_the_trainer_logged(client, dataset, tmp_path, monkeypatch):
    backend = FakeBackend([TrainingProgress(step=10, total_steps=20, loss=1.25), TrainingProgress(step=20, total_steps=20, loss=0.31)])
    monkeypatch.setattr(api, "build_training_backend", lambda: backend)

    started = _start(client, dataset, tmp_path)
    job = _wait(client, started.json()["job_id"])

    assert started.status_code == 202
    assert job["status"] == "completed"
    assert job["loss_history"] == [1.25, 0.31]
    assert (job["current_step"], job["total_steps"], job["current_loss"]) == (20, 20, 0.31)
    assert job["progress_percent"] == 100
    assert job["checkpoint_dir"] == str(tmp_path / "ckpt")


def test_a_backend_that_cannot_train_fails_the_job_without_inventing_progress(client, dataset, tmp_path, monkeypatch):
    error = RuntimeError("Training dependencies missing. Install via `pip install 'autotune[training]'`.")
    monkeypatch.setattr(api, "build_training_backend", lambda: FakeBackend(error=error))

    job = _wait(client, _start(client, dataset, tmp_path).json()["job_id"])

    assert job["status"] == "failed"
    assert "Training dependencies missing" in job["error"]
    assert job["loss_history"] == [] and job["current_loss"] is None
    assert not (tmp_path / "ckpt" / "autotune_meta.json").exists()


def test_a_missing_dataset_is_rejected_before_any_job_starts(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "build_training_backend", lambda: pytest.fail("no backend should be built"))

    response = _start(client, tmp_path / "missing.jsonl", tmp_path)

    assert response.status_code == 422


def test_an_unparseable_dataset_is_rejected(client, tmp_path, monkeypatch):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    monkeypatch.setattr(api, "build_training_backend", lambda: pytest.fail("no backend should be built"))

    assert _start(client, bad, tmp_path).status_code == 422


def test_cancelling_a_job_asks_the_backend_to_stop(client, dataset, tmp_path, monkeypatch):
    release = threading.Event()
    backend = FakeBackend(release=release)
    monkeypatch.setattr(api, "build_training_backend", lambda: backend)
    job_id = _start(client, dataset, tmp_path).json()["job_id"]

    response = client.post(f"/api/training/cancel/{job_id}")
    release.set()
    job = _wait(client, job_id)

    assert response.status_code == 200
    assert backend.cancelled == [job_id]
    assert job["status"] == "cancelled"


def test_the_simulated_mock_mode_is_gone(client, dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "build_training_backend", lambda: FakeBackend(error=RuntimeError("no GPU")))

    job = _wait(client, _start(client, dataset, tmp_path, mock=True).json()["job_id"])

    assert job["status"] == "failed"
    assert job["loss_history"] == []


def test_an_unknown_job_is_404(client):
    assert client.get("/api/training/status/nope").status_code == 404
    assert client.post("/api/training/cancel/nope").status_code == 404
