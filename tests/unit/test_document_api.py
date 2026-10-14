"""Uploads persist generated splits and expose progress through the job store."""
import json
import time
from unittest.mock import Mock

from fastapi.testclient import TestClient
from autotune.presentation.api import app as api
from autotune.infrastructure.evaluators.openai_client import OpenAIChatClient


def test_document_upload_and_training_reuses_generated_records(monkeypatch, isolated_data_dir):
    monkeypatch.setenv("AUTOTUNE_DOCUMENT_MODEL", "local")
    replies = iter(['[{"question":"Describe the safety policy", "answer":"The safety policy requires daily review."}]', '{"supported": true}'])
    monkeypatch.setattr(OpenAIChatClient, "complete", lambda *a, **kw: next(replies))
    client = TestClient(api.create_app())
    uploaded = client.post("/api/dataset/upload", files={"file": ("policy.md", "The safety policy requires daily review.")})
    assert uploaded.status_code == 200, uploaded.text
    assert uploaded.json()["progress"]["pairs_kept"] == 1
    backend = Mock()
    backend.train.side_effect = RuntimeError("stop before GPU")
    monkeypatch.setattr(api, "build_training_backend", lambda: backend)
    response = client.post("/api/training/start", json={"dataset_name": "policy.md"})
    assert response.status_code == 202, response.text
    for _ in range(100):
        state = client.get('/api/training/status/' + response.json()['job_id']).json()
        if state['status'] == 'failed': break
        time.sleep(.01)
    assert state['error'] == 'stop before GPU'
    assert (isolated_data_dir / "uploads" / "policy.md.cleaned.json").exists()


def test_background_upload_exposes_generator_failure(monkeypatch):
    monkeypatch.setenv("AUTOTUNE_DOCUMENT_MODEL", "")
    client = TestClient(api.create_app())
    started = client.post('/api/dataset/upload?background=true', files={'file': ('policy.md', 'text')})
    assert started.status_code == 202
    for _ in range(100):
        state = client.get('/api/dataset/status/' + started.json()['job_id']).json()
        if state['status'] == 'failed': break
        time.sleep(.01)
    assert state['status'] == 'failed'
    assert 'AUTOTUNE_DOCUMENT_MODEL' in state['error']
    assert state.get('result') is None


def test_missing_upload_job_is_not_success():
    assert TestClient(api.create_app()).get('/api/dataset/status/missing').status_code == 404
