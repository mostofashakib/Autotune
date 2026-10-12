"""The settings endpoint reports which credentials are configured, never their values."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autotune import settings
from autotune.presentation.api import app as api


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return TestClient(api.create_app())


def test_every_credential_is_listed_with_its_purpose(client):
    rows = {row["name"]: row for row in client.get("/api/settings/credentials").json()}

    assert set(rows) == {"RUNPOD_API_KEY", "HF_TOKEN"}
    assert rows["RUNPOD_API_KEY"]["purpose"] == settings.RUNPOD_API_KEY.purpose
    assert rows["RUNPOD_API_KEY"]["configured"] is False


def test_a_configured_credential_is_reported_without_its_value(client, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret-value")

    response = client.get("/api/settings/credentials")

    rows = {row["name"]: row for row in response.json()}
    assert rows["RUNPOD_API_KEY"]["configured"] is True
    assert "rp-secret-value" not in response.text


def test_the_placeholder_counts_as_not_configured(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "mock_key")

    rows = {row["name"]: row for row in client.get("/api/settings/credentials").json()}

    assert rows["HF_TOKEN"]["configured"] is False


def test_the_env_file_location_is_reported(client):
    body = client.get("/api/settings/credentials")

    assert body.status_code == 200
    assert all(row["env_file"] == ".env" for row in body.json())
