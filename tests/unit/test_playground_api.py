"""The playground returns the model's real answer, or fails. It never writes canned text."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autotune.infrastructure.evaluators.openai_client import ModelCallError
from autotune.presentation.api import app as api

ENDPOINT = {"base_url": "http://localhost:11434/v1", "model": "gemma3:1b"}


class FakeClient:
    def __init__(self, reply="4", error=None):
        self.reply, self.error = reply, error
        self.calls = []

    def complete(self, endpoint, messages, *, temperature=0.7, max_tokens=1024):
        self.calls.append((endpoint.model, messages, max_tokens))
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def client():
    return TestClient(api.create_app())


def test_the_playground_returns_the_models_answer(client, monkeypatch):
    fake = FakeClient("4")
    monkeypatch.setattr(api, "OpenAIChatClient", lambda: fake)

    response = client.post("/api/playground/generate", json={"prompt": "2+2?", "model": ENDPOINT})

    assert response.status_code == 200
    assert response.json()["response"] == "4"
    assert fake.calls == [("gemma3:1b", [{"role": "user", "content": "2+2?"}], 1024)]


def test_a_model_that_does_not_answer_is_reported(client, monkeypatch):
    monkeypatch.setattr(api, "OpenAIChatClient", lambda: FakeClient(error=ModelCallError("gemma3:1b at http://localhost:11434/v1 did not answer")))

    response = client.post("/api/playground/generate", json={"prompt": "hi", "model": ENDPOINT})

    assert response.status_code == 502
    assert "did not answer" in response.json()["detail"]


def test_an_empty_prompt_is_rejected(client):
    assert client.post("/api/playground/generate", json={"prompt": "  ", "model": ENDPOINT}).status_code == 422


def test_a_request_without_a_model_endpoint_is_rejected(client):
    assert client.post("/api/playground/generate", json={"prompt": "hi", "model_type": "trained"}).status_code == 422
