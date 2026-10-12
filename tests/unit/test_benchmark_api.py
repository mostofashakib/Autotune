"""The benchmark API runs a real comparison in the background and never invents numbers."""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from autotune.core.domain.models import SampleResult
from autotune.infrastructure.evaluators.openai_client import ModelCallError
from autotune.presentation.api import app as api


class FakeRunner:
    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on

    def run(self, model):
        if model.model == self.fail_on:
            raise ModelCallError(f"{model.model} at {model.base_url} did not answer")
        return [SampleResult(task_id="t", passed=model.model == "trained", score=1.0 if model.model == "trained" else 0.0)]


@pytest.fixture
def client():
    return TestClient(api.create_app())


def _body(tmp_path, **overrides):
    body = {
        "mode": "dataset", "path": str(tmp_path / "sft.jsonl"), "k": 1,
        "base": {"base_url": "http://b/v1", "model": "base"},
        "trained": {"base_url": "http://t/v1", "model": "trained"},
        "output_dir": str(tmp_path / "out"),
    }
    return {**body, **overrides}


def _wait(client, job_id):
    for _ in range(200):
        job = client.get(f"/api/benchmark/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.01)
    raise AssertionError("benchmark job never finished")


def test_a_benchmark_job_reports_the_measured_comparison(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "build_benchmark_runner", lambda request: FakeRunner())

    started = client.post("/api/benchmark", json=_body(tmp_path))
    job = _wait(client, started.json()["job_id"])

    assert started.status_code == 202
    assert job["status"] == "completed"
    assert job["report"]["delta_pass_at_1"] == 1.0
    assert json.loads((tmp_path / "out" / "benchmark_report.json").read_text())["mode"] == "dataset"


def test_a_model_failure_fails_the_job_with_no_report(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "build_benchmark_runner", lambda request: FakeRunner(fail_on="trained"))

    job = _wait(client, client.post("/api/benchmark", json=_body(tmp_path)).json()["job_id"])

    assert job["status"] == "failed"
    assert "did not answer" in job["error"]
    assert job["report"] is None
    assert not (tmp_path / "out").exists()


def test_a_request_that_cannot_build_a_runner_is_rejected(client, tmp_path):
    response = client.post("/api/benchmark", json=_body(tmp_path, path=str(tmp_path / "missing.jsonl")))

    assert response.status_code == 422


def test_a_judge_is_only_accepted_for_dataset_mode(client, tmp_path):
    judge = {"base_url": "http://j/v1", "model": "judge"}

    response = client.post("/api/benchmark", json=_body(tmp_path, mode="harbor", judge=judge))

    assert response.status_code == 422


def test_an_unknown_job_is_404(client):
    assert client.get("/api/benchmark/nope").status_code == 404


def test_the_old_evaluate_endpoint_with_canned_scores_is_gone(client):
    assert client.post("/api/evaluate", json={}).status_code in (404, 405)
