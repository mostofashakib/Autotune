"""The serving API starts vLLM for real and reports its actual state."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autotune.core.domain.models import ServingEndpoint, ServingStatus
from autotune.presentation.api import app as api


class FakeVLLM:
    def __init__(self, start_error: Exception | None = None, status: str = "starting"):
        self.start_error, self.status = start_error, status
        self.started, self.stopped = [], False

    def start(self, model_path, model_name="", port=8000, gpu_memory_utilization=0.85, log_path=None):
        self.started.append((str(model_path), port))
        if self.start_error:
            raise self.start_error
        return ServingEndpoint(model_name="m", base_url=f"http://127.0.0.1:{port}", port=port)

    def state(self):
        return ServingStatus(status=self.status, base_url="http://127.0.0.1:8000", log_tail="")

    def stop(self):
        self.stopped = True


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_ACTIVE_SERVING", None)
    return TestClient(api.create_app())


def test_starting_serving_launches_vllm_and_reports_it_as_starting(client, monkeypatch):
    fake = FakeVLLM()
    monkeypatch.setattr(api, "VLLMServingProvider", lambda: fake)

    response = client.post("/api/serve/start", json={"model_path": "./checkpoints/m", "port": 8000})

    assert response.status_code == 202
    assert response.json()["status"] == "starting"
    assert response.json()["endpoint_url"] == "http://127.0.0.1:8000/v1"
    assert fake.started == [("./checkpoints/m", 8000)]


def test_a_vllm_that_cannot_be_spawned_is_reported(client, monkeypatch):
    monkeypatch.setattr(api, "VLLMServingProvider", lambda: FakeVLLM(start_error=RuntimeError("Could not spawn vLLM server: python not found")))

    response = client.post("/api/serve/start", json={"model_path": "m"})

    assert response.status_code == 503
    assert "Could not spawn" in response.json()["detail"]


def test_status_is_the_servers_real_state(client, monkeypatch):
    fake = FakeVLLM(status="online")
    monkeypatch.setattr(api, "VLLMServingProvider", lambda: fake)
    client.post("/api/serve/start", json={"model_path": "m"})

    assert client.get("/api/serve/status").json()["status"] == "online"


def test_status_without_a_server_is_stopped(client):
    assert client.get("/api/serve/status").json()["status"] == "stopped"


def test_stopping_terminates_the_server(client, monkeypatch):
    fake = FakeVLLM()
    monkeypatch.setattr(api, "VLLMServingProvider", lambda: fake)
    client.post("/api/serve/start", json={"model_path": "m"})

    response = client.post("/api/serve/stop")

    assert response.status_code == 200
    assert fake.stopped
