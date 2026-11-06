"""The failures a user can cause or fix, and the HTTP status each one maps to."""
from __future__ import annotations

from autotune.core.ports.sanitizer import IngestionError
from autotune.infrastructure.evaluators.harbor_benchmark import HarborError
from autotune.infrastructure.evaluators.judge import JudgeError
from autotune.infrastructure.evaluators.openai_client import ModelCallError
from autotune.infrastructure.sanitizers.dataset_loader import MalformedExportError
from autotune.settings import CredentialNotConfigured, EndpointKeyMissing

EXPECTED_FAILURES: tuple[type[Exception], ...] = (
    IngestionError,
    ValueError,
    CredentialNotConfigured,
    ModelCallError,
    JudgeError,
    HarborError,
    MalformedExportError,
    EndpointKeyMissing,
    OSError,
)

_UPSTREAM_FAILURES = (ModelCallError, JudgeError, HarborError, ConnectionError)


def http_status(exc: Exception) -> int:
    """422 for bad input, 503 for a missing credential, 502 for a failing upstream, else 500."""
    if isinstance(exc, CredentialNotConfigured):
        return 503
    if isinstance(exc, _UPSTREAM_FAILURES):
        return 502
    if isinstance(exc, ValueError):
        return 422
    return 500
