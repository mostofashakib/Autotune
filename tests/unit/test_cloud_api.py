"""Cloud GPU endpoints report RunPod's real answers or fail. They never invent a catalog or a pod."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autotune import settings
from autotune.core.domain.models import RunPodInstance
from autotune.presentation.api import app as api


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    return TestClient(api.create_app())


class FakeRunPod:
    def __init__(self, error: Exception | None = None, stopped: bool = True):
        self.error, self.stopped = error, stopped
        self.calls = []

    def _maybe_fail(self):
        if self.error:
            raise self.error

    def list_available_gpus(self):
        self.calls.append("gpus")
        self._maybe_fail()
        return [{"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "memory_gb": 24, "community_price": 0.34}]

    def create_instance(self, config):
        self.calls.append(("launch", config.gpu_type_id))
        self._maybe_fail()
        return RunPodInstance(pod_id="pod_real", name=config.pod_name, status="RUNNING", gpu_name=config.gpu_type_id)

    def stop_instance(self, pod_id):
        self.calls.append(("stop", pod_id))
        self._maybe_fail()
        return self.stopped


@pytest.fixture
def runpod(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-live-123")
    fake = FakeRunPod()
    monkeypatch.setattr(api, "RunPodCloudProvider", lambda api_key=None: fake)
    return fake


def test_without_a_configured_key_the_catalog_is_refused_and_names_the_variable(client):
    response = client.get("/api/cloud/gpus")

    assert response.status_code == 503
    assert "RUNPOD_API_KEY" in response.json()["detail"]


def test_without_a_configured_key_no_pod_is_invented(client):
    response = client.post("/api/cloud/launch", json={"gpu_type": "NVIDIA GeForce RTX 4090"})

    assert response.status_code == 503
    assert "RUNPOD_API_KEY" in response.json()["detail"]


def test_the_catalog_is_runpods_answer(client, runpod):
    response = client.get("/api/cloud/gpus")

    assert response.status_code == 200
    assert response.json()[0]["community_price"] == 0.34


def test_a_runpod_failure_is_reported_instead_of_a_stand_in_catalog(client, runpod):
    runpod.error = RuntimeError("RunPod GraphQL error: unauthorized")

    response = client.get("/api/cloud/gpus")

    assert response.status_code == 502
    assert "unauthorized" in response.json()["detail"]


def test_a_launched_pod_is_the_one_runpod_created(client, runpod):
    response = client.post("/api/cloud/launch", json={"gpu_type": "NVIDIA GeForce RTX 4090", "pod_name": "trainer"})

    assert response.status_code == 200
    assert response.json()["pod_id"] == "pod_real"
    assert runpod.calls == [("launch", "NVIDIA GeForce RTX 4090")]


def test_a_failed_launch_is_reported_instead_of_a_simulated_pod(client, runpod):
    runpod.error = RuntimeError("no capacity for RTX 4090")

    response = client.post("/api/cloud/launch", json={"gpu_type": "NVIDIA GeForce RTX 4090"})

    assert response.status_code == 502
    assert "no capacity" in response.json()["detail"]


def test_stopping_a_pod_terminates_it_on_runpod(client, runpod):
    response = client.post("/api/cloud/stop/pod_real")

    assert response.status_code == 200
    assert runpod.calls == [("stop", "pod_real")]


def test_a_pod_runpod_would_not_terminate_is_reported(client, runpod):
    runpod.stopped = False

    assert client.post("/api/cloud/stop/pod_real").status_code == 502
