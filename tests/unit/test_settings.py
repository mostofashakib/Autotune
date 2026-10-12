"""Credentials are read in one place, and a placeholder never counts as configured."""
from __future__ import annotations

import os

import pytest

from autotune import settings
from autotune.settings import (
    HF_TOKEN,
    PLACEHOLDER,
    RUNPOD_API_KEY,
    CredentialNotConfigured,
    credential_value,
    export_configured_credentials,
    is_configured,
    require_credential,
)


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    monkeypatch.setattr(settings, "ENV_FILE", path)
    for name in ("RUNPOD_API_KEY", "HF_TOKEN"):
        # Set, then delete, so monkeypatch restores whatever the test exports.
        monkeypatch.setenv(name, "recorded")
        monkeypatch.delenv(name)
    return path


def test_an_unset_credential_is_the_placeholder_and_not_configured(env_file):
    assert credential_value(RUNPOD_API_KEY) == PLACEHOLDER == "mock_key"
    assert not is_configured(credential_value(RUNPOD_API_KEY))


def test_the_env_file_supplies_a_credential(env_file):
    env_file.write_text('# keys\nRUNPOD_API_KEY="rp-live-123"\nexport HF_TOKEN=hf_abc\n')

    assert credential_value(RUNPOD_API_KEY) == "rp-live-123"
    assert credential_value(HF_TOKEN) == "hf_abc"


def test_the_process_environment_wins_over_the_env_file(env_file, monkeypatch):
    env_file.write_text("RUNPOD_API_KEY=from-file\n")
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")

    assert credential_value(RUNPOD_API_KEY) == "from-env"


def test_an_explicit_override_wins_over_both(env_file, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")

    assert credential_value(RUNPOD_API_KEY, override="from-request") == "from-request"


def test_requiring_an_unconfigured_credential_names_the_variable_and_its_purpose(env_file):
    env_file.write_text(f"RUNPOD_API_KEY={PLACEHOLDER}\n")

    with pytest.raises(CredentialNotConfigured, match="RUNPOD_API_KEY") as caught:
        require_credential(RUNPOD_API_KEY)
    assert "RunPod" in str(caught.value)


def test_exporting_skips_placeholders_so_libraries_never_send_a_fake_token(env_file, monkeypatch):
    env_file.write_text(f"HF_TOKEN={PLACEHOLDER}\nRUNPOD_API_KEY=rp-live-123\n")

    export_configured_credentials()

    assert "HF_TOKEN" not in os.environ
    assert os.environ["RUNPOD_API_KEY"] == "rp-live-123"


def test_the_example_env_file_lists_every_credential_at_its_placeholder():
    example = (settings.ROOT / ".env.example").read_text()

    for credential in settings.CREDENTIALS:
        assert f"{credential.env_var}={PLACEHOLDER}" in example


def test_the_api_exports_configured_credentials_at_startup(env_file):
    from autotune.presentation.api.app import create_app

    env_file.write_text("HF_TOKEN=hf_live_abc\n")

    create_app()

    assert os.environ["HF_TOKEN"] == "hf_live_abc"


def test_nothing_outside_settings_reads_a_credential_from_the_environment():
    sources = [p for p in (settings.ROOT / "src").rglob("*.py") if p.name != "settings.py"]
    offenders = [
        str(p.relative_to(settings.ROOT)) for p in sources
        for credential in settings.CREDENTIALS
        if f'environ.get("{credential.env_var}"' in p.read_text() or f'"{credential.env_var}", "' in p.read_text()
    ]
    assert offenders == []
