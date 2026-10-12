"""Unit tests for Autotune FastAPI Web UI backend."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from autotune.presentation.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_index_page_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AUTOTUNE" in response.text
    assert "Training Cockpit" in response.text


def test_system_status(client):
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "vram_gb" in data
    assert "cuda_available" in data
    assert "device" in data


def test_load_sample_dataset(client):
    response = client.post("/api/dataset/sample?task=rl_grpo")
    assert response.status_code == 200
    data = response.json()
    assert data["task_type"] == "rl_grpo"
    assert data["total_records"] > 0
    assert "recipe" in data
    assert data["recipe"]["objective"] == "grpo"


def test_playground_generate(client):
    payload = {"prompt": "Solve optimization task #1", "model_type": "trained"}
    response = client.post("/api/playground/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["response"]) > 0


def test_cloud_gpus_catalog(client):
    response = client.get("/api/cloud/gpus")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "NVIDIA" in data[0]["id"]
