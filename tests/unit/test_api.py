"""Unit tests for Autotune FastAPI Web UI backend."""
from __future__ import annotations

import json

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


def test_there_is_no_generated_sample_dataset(client):
    assert client.post("/api/dataset/sample?task=rl_grpo").status_code in (404, 405)


SFT_ROWS = "\n".join(
    json.dumps({"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]})
    for i in range(12)
) + "\n"


def test_an_upload_is_saved_in_the_data_dir_and_inspected(client, isolated_data_dir):
    response = client.post("/api/dataset/upload", files={"file": ("sft.jsonl", SFT_ROWS)})

    assert response.status_code == 200
    body = response.json()
    assert body["file_name"] == "sft.jsonl"
    assert body["total_records"] == 12
    assert (isolated_data_dir / "uploads" / "sft.jsonl").read_text() == SFT_ROWS


@pytest.mark.parametrize("name", ["../escape.jsonl", "/etc/passwd", "nested/dir/x.jsonl", "..", ""])
def test_an_upload_name_that_is_a_path_is_rejected(client, isolated_data_dir, name):
    response = client.post("/api/dataset/upload", files={"file": (name, SFT_ROWS)})

    assert response.status_code in (400, 422)
    assert not (isolated_data_dir.parent / "escape.jsonl").exists()
    assert not list((isolated_data_dir / "uploads").glob("*")) if (isolated_data_dir / "uploads").exists() else True


def test_an_unparseable_upload_is_rejected_and_not_kept(client, isolated_data_dir):
    response = client.post("/api/dataset/upload", files={"file": ("broken.jsonl", "not json at all\n")})

    assert response.status_code == 400
    assert not (isolated_data_dir / "uploads" / "broken.jsonl").exists()


def test_training_finds_an_uploaded_dataset_by_its_name(client, monkeypatch):
    from autotune.presentation.api import app as api

    seen = {}

    class Backend:
        def train(self, job, dataset, on_progress=None):
            seen["path"] = job.dataset_path
            raise RuntimeError("stop here")

        def cancel(self, job_id):
            pass

    monkeypatch.setattr(api, "build_training_backend", lambda: Backend())
    client.post("/api/dataset/upload", files={"file": ("sft.jsonl", SFT_ROWS)})

    response = client.post("/api/training/start", json={"dataset_name": "sft.jsonl"})

    assert response.status_code == 202
    assert response.json()["job_id"]
