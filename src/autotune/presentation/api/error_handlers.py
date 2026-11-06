"""One error policy for every route."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from autotune.presentation.errors import STATUS_BY_FAILURE, http_status

logger = logging.getLogger(__name__)


def _expected_failure(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=http_status(exc), content={"detail": str(exc)})


def _unexpected_failure(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal error. See the server log."})


def install_error_handlers(app: FastAPI) -> None:
    for failure, _ in STATUS_BY_FAILURE:
        app.add_exception_handler(failure, _expected_failure)
    app.add_exception_handler(Exception, _unexpected_failure)
