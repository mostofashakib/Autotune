"""The failures a user can cause or fix, and the HTTP status each one maps to."""
from __future__ import annotations

from autotune.core.domain.errors import HardwareUnavailable, MissingExtraError, ProviderError
from autotune.core.ports.sanitizer import IngestionError
from autotune.infrastructure.evaluators.harbor_benchmark import HarborError
from autotune.infrastructure.evaluators.judge import JudgeError
from autotune.infrastructure.evaluators.openai_client import ModelCallError
from autotune.infrastructure.sanitizers.dataset_loader import MalformedExportError
from autotune.settings import CredentialNotConfigured, EndpointKeyMissing

# First match wins. Classes listed here get a handler that returns their message.
STATUS_BY_FAILURE: tuple[tuple[type[Exception], int], ...] = (
    (CredentialNotConfigured, 503),
    (MissingExtraError, 503),
    (HardwareUnavailable, 503),
    (ProviderError, 502),
    (ModelCallError, 502),
    (JudgeError, 502),
    (HarborError, 502),
    (ConnectionError, 502),
    (IngestionError, 422),
    (MalformedExportError, 422),
    (EndpointKeyMissing, 422),
    (FileNotFoundError, 422),
    (ValueError, 422),
)

# Other OSErrors (PermissionError and the rest) are expected by workers and the CLI,
# but the API answers them with the generic 500 so raw paths never reach a client.
EXPECTED_FAILURES: tuple[type[Exception], ...] = (*(failure for failure, _ in STATUS_BY_FAILURE), OSError)


def http_status(exc: Exception) -> int:
    """422 for bad input, 503 for a missing credential, extra, or GPU, 502 for a failing upstream, else 500."""
    for failure, status in STATUS_BY_FAILURE:
        if isinstance(exc, failure):
            return status
    return 500
