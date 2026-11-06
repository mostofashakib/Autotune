import logging

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from autotune.core.domain.errors import MissingExtraError
from autotune.core.ports.sanitizer import IngestionError
from autotune.infrastructure.evaluators.harbor_benchmark import HarborError
from autotune.infrastructure.evaluators.judge import JudgeError
from autotune.infrastructure.evaluators.openai_client import ModelCallError
from autotune.infrastructure.sanitizers.dataset_loader import MalformedExportError
from autotune.presentation.api.error_handlers import install_error_handlers
from autotune.presentation.errors import EXPECTED_FAILURES, http_status
from autotune.settings import CredentialNotConfigured, EndpointKeyMissing

STATUS_BY_FAILURE = [
    (IngestionError("bad rows"), 422),
    (ValueError("bad value"), 422),
    (MalformedExportError("bad export"), 422),
    (EndpointKeyMissing("no endpoint key"), 422),
    (CredentialNotConfigured("no key"), 503),
    (ModelCallError("model down"), 502),
    (JudgeError("judge down"), 502),
    (HarborError("harbor down"), 502),
    (ConnectionError("refused"), 502),
    (MissingExtraError("Run: uv sync --extra documents"), 503),
    (FileNotFoundError("missing"), 422),
]


class Body(BaseModel):
    count: int


def _client(raised: Exception) -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise raised

    @app.get("/forbidden")
    def forbidden() -> None:
        raise HTTPException(status_code=403, detail="no")

    @app.post("/body")
    def body(payload: Body) -> dict:
        return {"count": payload.count}

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("failure,status", STATUS_BY_FAILURE)
def test_an_expected_failure_gets_its_mapped_status_and_message(failure, status):
    response = _client(failure).get("/boom")

    assert response.status_code == status
    assert response.json() == {"detail": str(failure)}


@pytest.mark.parametrize("failure,status", STATUS_BY_FAILURE)
def test_http_status_matches_the_handlers(failure, status):
    assert http_status(failure) == status
    assert isinstance(failure, EXPECTED_FAILURES)


def test_an_unexpected_failure_is_logged_and_hidden(caplog):
    with caplog.at_level(logging.ERROR):
        response = _client(KeyError("secret detail")).get("/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error. See the server log."}
    assert "secret detail" in caplog.text


def test_an_unexpected_os_error_hides_its_paths(caplog):
    with caplog.at_level(logging.ERROR):
        response = _client(PermissionError("/etc/shadow")).get("/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error. See the server log."}
    assert "/etc/shadow" in caplog.text


def test_http_status_defaults_to_500():
    assert http_status(KeyError("x")) == 500


def test_http_exceptions_keep_their_own_status():
    response = _client(ValueError("unused")).get("/forbidden")

    assert response.status_code == 403
    assert response.json() == {"detail": "no"}


def test_a_bad_request_body_keeps_fastapis_validation_shape():
    response = _client(ValueError("unused")).post("/body", json={"count": "many"})

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert response.json()["detail"][0]["loc"] == ["body", "count"]
