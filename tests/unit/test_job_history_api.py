"""Training and benchmark runs are listed, and outlive the server that ran them."""
from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from autotune import settings
from autotune.core.domain.models import CheckpointResult, JobStatus, SampleResult, TrainingProgress
from autotune.infrastructure.storage.job_store import JobStore
from autotune.presentation.api import app as api


class FakeBackend:
    def __init__(self, release: threading.Event | None = None):
        self.release = release

    def train(self, job, dataset, on_progress=None):
        on_progress(TrainingProgress(step=1, total_steps=2, loss=0.9))
        if self.release is not None:
            self.release.wait(5)
        return CheckpointResult(job_id=job.id, checkpoint_dir=job.output_dir, final_loss=0.4, total_steps=2, status=JobStatus.COMPLETED)

    def cancel(self, job_id):
        pass


class FakeRunner:
    def run(self, model):
        return [SampleResult(task_id="t1", passed=model.model == "trained", score=1.0 if model.model == "trained" else 0.0)]


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "sft.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(12)]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _wait(client, url):
    for _ in range(300):
        body = client.get(url).json()
        if body["status"] != "running":
            return body
        time.sleep(0.01)
    raise AssertionError(f"{url} never finished")


def _train(client, dataset, tmp_path):
    body = {"dataset_name": str(dataset), "base_model": "Qwen/Qwen2.5-0.5B-Instruct", "output_dir": str(tmp_path / "ckpt")}
    return client.post("/api/training/start", json=body).json()["job_id"]


def _benchmark(client, tmp_path):
    body = {
        "mode": "dataset", "path": "held_out.jsonl", "k": 1, "output_dir": str(tmp_path / "out"),
        "base": {"base_url": "http://localhost:1/v1", "model": "base"},
        "trained": {"base_url": "http://localhost:2/v1", "model": "trained"},
    }
    return client.post("/api/benchmark", json=body).json()["job_id"]


def test_a_finished_training_run_survives_a_restart(monkeypatch, dataset, tmp_path):
    monkeypatch.setattr(api, "build_training_backend", lambda: FakeBackend())
    first = TestClient(api.create_app())
    job_id = _train(first, dataset, tmp_path)
    finished = _wait(first, f"/api/training/status/{job_id}")

    restarted = TestClient(api.create_app())
    body = restarted.get(f"/api/training/status/{job_id}").json()

    assert body["status"] == "completed"
    assert body["logs"] == finished["logs"]
    assert body["loss_history"] == [0.9]


def test_the_training_list_shows_runs_newest_first_without_their_logs(monkeypatch, dataset, tmp_path):
    monkeypatch.setattr(api, "build_training_backend", lambda: FakeBackend())
    client = TestClient(api.create_app())
    first = _train(client, dataset, tmp_path)
    _wait(client, f"/api/training/status/{first}")
    second = _train(client, dataset, tmp_path)
    _wait(client, f"/api/training/status/{second}")

    runs = client.get("/api/training").json()

    assert [run["job_id"] for run in runs] == [second, first]
    assert runs[0]["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert runs[0]["dataset_name"] == "sft.jsonl"
    assert runs[0]["created_at"]
    assert "logs" not in runs[0] and "loss_history" not in runs[0] and "backend" not in runs[0]


def test_a_run_cut_off_by_a_restart_shows_as_interrupted(monkeypatch, dataset, tmp_path):
    release = threading.Event()
    monkeypatch.setattr(api, "build_training_backend", lambda: FakeBackend(release))
    first = TestClient(api.create_app())
    job_id = _train(first, dataset, tmp_path)

    body = TestClient(api.create_app()).get(f"/api/training/status/{job_id}").json()
    release.set()

    assert body["status"] == "interrupted"
    assert "restarted" in body["error"]


def test_cancelling_a_run_from_before_the_restart_is_a_conflict(isolated_data_dir):
    JobStore(isolated_data_dir / "autotune.db").save("training", "job_old", {"job_id": "job_old", "status": "completed"})

    response = TestClient(api.create_app()).post("/api/training/cancel/job_old")

    assert response.status_code == 409
    assert "completed" in response.json()["detail"]


def test_an_unknown_run_is_still_not_found():
    assert TestClient(api.create_app()).get("/api/training/status/job_nope").status_code == 404


def test_the_job_database_lives_in_the_data_dir(monkeypatch, dataset, tmp_path, isolated_data_dir):
    monkeypatch.setattr(api, "build_training_backend", lambda: FakeBackend())
    client = TestClient(api.create_app())
    _wait(client, f"/api/training/status/{_train(client, dataset, tmp_path)}")

    assert settings.data_dir() == isolated_data_dir
    assert JobStore(isolated_data_dir / "autotune.db").list("training")


def test_a_finished_benchmark_survives_a_restart_and_is_listed_with_its_headline(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "build_benchmark_runner", lambda request: FakeRunner())
    first = TestClient(api.create_app())
    job_id = _benchmark(first, tmp_path)
    _wait(first, f"/api/benchmark/{job_id}")

    restarted = TestClient(api.create_app())
    body = restarted.get(f"/api/benchmark/{job_id}").json()
    runs = restarted.get("/api/benchmark").json()

    assert body["status"] == "completed"
    assert body["report"]["delta_pass_at_1"] == 1.0
    assert runs[0]["job_id"] == job_id
    assert runs[0]["base_model"] == "base" and runs[0]["trained_model"] == "trained"
    assert runs[0]["headline"] == {"base_pass_at_1": 0.0, "trained_pass_at_1": 1.0, "delta_pass_at_1": 1.0}
    assert "report" not in runs[0] and "logs" not in runs[0]


def test_a_failed_benchmark_lists_without_a_headline(monkeypatch, tmp_path):
    class Broken:
        def run(self, model):
            raise ValueError("endpoint refused")

    monkeypatch.setattr(api, "build_benchmark_runner", lambda request: Broken())
    client = TestClient(api.create_app())
    _wait(client, f"/api/benchmark/{_benchmark(client, tmp_path)}")

    run = client.get("/api/benchmark").json()[0]

    assert run["status"] == "failed"
    assert run["headline"] is None
    assert "endpoint refused" in run["error"]
