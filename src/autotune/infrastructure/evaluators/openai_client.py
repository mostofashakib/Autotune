"""A minimal client for OpenAI-compatible chat completion servers (vLLM, Ollama, RunPod)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from autotune.core.domain.models import ModelEndpoint
from autotune.infrastructure.http import http_request
from autotune.settings import EndpointKeyMissing, endpoint_api_key

# (request, timeout seconds) -> response body
Transport = Callable[[urllib.request.Request, float], bytes]


class ModelCallError(RuntimeError):
    """A model endpoint could not produce an answer."""


def default_transport(request: urllib.request.Request, timeout: float) -> bytes:
    """The reply body of a 2xx answer. Any other status raises ConnectionError."""
    status, body = http_request(
        request.get_method(), request.full_url, headers=dict(request.header_items()), body=request.data, timeout=timeout,
    )
    if not 200 <= status < 300:
        raise ConnectionError(f"HTTP request to {request.full_url} returned status {status}: {body[:200]!r}")
    return body


class OpenAIChatClient:
    def __init__(self, transport: Transport | None = None, timeout_s: float = 120.0) -> None:
        self._transport = transport or default_transport
        self._timeout_s = timeout_s

    def complete(
        self, endpoint: ModelEndpoint, messages: list[dict[str, str]],
        *, temperature: float = 0.7, max_tokens: int = 1024,
    ) -> str:
        """The assistant text for `messages`. Raises ModelCallError on any failure."""
        where = f"{endpoint.model} at {endpoint.base_url}"
        headers = {"Content-Type": "application/json", **_auth_header(endpoint)}
        body = {"model": endpoint.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        request = urllib.request.Request(
            endpoint.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), headers=headers, method="POST",
        )
        try:
            raw = self._transport(request, self._timeout_s)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelCallError(f"{where} did not answer: {exc}") from exc
        return _content(raw, where)


def _auth_header(endpoint: ModelEndpoint) -> dict[str, str]:
    try:
        key = endpoint_api_key(endpoint)
    except EndpointKeyMissing as exc:
        raise ModelCallError(str(exc)) from exc
    return {"Authorization": f"Bearer {key}"} if key else {}


def _content(raw: bytes, where: str) -> str:
    try:
        payload: Any = json.loads(raw)
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ModelCallError(f"{where} sent a malformed reply: {raw[:200]!r}") from exc
    if not isinstance(content, str):
        raise ModelCallError(f"{where} sent a malformed reply: content is {type(content).__name__}")
    # Reasoning models can spend the whole budget thinking and return no answer.
    if not content.strip() and choice.get("finish_reason") == "length":
        raise ModelCallError(f"{where} ran out of tokens before answering. Raise max_tokens.")
    return content
