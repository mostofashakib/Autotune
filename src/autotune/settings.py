"""Every credential Autotune needs, read in one place.

A value comes from an explicit override (a request or CLI flag), then the
process environment, then `.env` at the project root. A credential left at its
placeholder counts as not configured: features that need it refuse to run and
name the variable, instead of pretending to work.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from autotune.core.domain.models import ModelEndpoint

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
PLACEHOLDER = "mock_key"


DEFAULT_UI_ORIGIN = "http://localhost:3001"


def ui_origin() -> str:
    """The web UI's origin, the only one the API accepts cross-origin requests from."""
    return os.environ.get("AUTOTUNE_UI_ORIGIN", "").strip() or DEFAULT_UI_ORIGIN


def data_dir() -> Path:
    """Where uploads and the job database live: AUTOTUNE_DATA_DIR, else <repo>/data."""
    configured = os.environ.get("AUTOTUNE_DATA_DIR", "").strip()
    return Path(configured) if configured else ROOT / "data"


@dataclass(frozen=True)
class Credential:
    env_var: str
    purpose: str


RUNPOD_API_KEY = Credential("RUNPOD_API_KEY", "renting cloud GPUs on RunPod")
HF_TOKEN = Credential("HF_TOKEN", "downloading gated Hugging Face base models")
CREDENTIALS = (RUNPOD_API_KEY, HF_TOKEN)


class CredentialNotConfigured(RuntimeError):
    """A feature needs a credential that is unset or still at its placeholder."""


def credential_value(credential: Credential, override: str | None = None) -> str:
    """The credential's value, or the placeholder when nothing sets it."""
    for value in (override, os.environ.get(credential.env_var), _env_file_values().get(credential.env_var)):
        if value and value.strip():
            return value.strip()
    return PLACEHOLDER


def is_configured(value: str) -> bool:
    return bool(value.strip()) and value.strip() != PLACEHOLDER


def require_credential(credential: Credential, override: str | None = None) -> str:
    value = credential_value(credential, override)
    if not is_configured(value):
        raise CredentialNotConfigured(
            f"{credential.env_var} is not configured. Set it in {ENV_FILE.name} to enable {credential.purpose}."
        )
    return value


def export_configured_credentials() -> None:
    """Put configured `.env` credentials into the environment for libraries that read it.

    Placeholders are skipped, so a library such as huggingface_hub never sends a fake token.
    """
    for credential in CREDENTIALS:
        value = credential_value(credential)
        if is_configured(value) and credential.env_var not in os.environ:
            os.environ[credential.env_var] = value


def _env_file_values() -> dict[str, str]:
    """KEY=VALUE pairs from ENV_FILE. Comments, blank lines, and `export` prefixes are allowed."""
    if not ENV_FILE.is_file():
        return {}
    values = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def setting_value(name: str, default: str = "") -> str:
    """Read a non-secret setting with environment precedence over .env."""
    return os.environ.get(name, _env_file_values().get(name, default)).strip()


def document_endpoint() -> ModelEndpoint:
    """The configured document generator. A model name is required even for Ollama."""
    model = setting_value("AUTOTUNE_DOCUMENT_MODEL")
    if not model or model == PLACEHOLDER:
        raise ValueError("Set AUTOTUNE_DOCUMENT_MODEL to a model served by AUTOTUNE_DOCUMENT_ENDPOINT (default http://localhost:11434/v1).")
    key_env = setting_value("AUTOTUNE_DOCUMENT_API_KEY_ENV") or None
    if key_env:
        key = require_credential(Credential(key_env, "document generation"))
        os.environ[key_env] = key
    return ModelEndpoint(base_url=setting_value("AUTOTUNE_DOCUMENT_ENDPOINT", "http://localhost:11434/v1"), model=model, api_key_env=key_env)
