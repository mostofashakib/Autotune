"""A minimal client for OpenAI-compatible chat completion servers (vLLM, Ollama, RunPod)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from autotune.core.domain.models import ModelEndpoint

# (request, timeout seconds) -> response body
Transport = Callable[[urllib.request.Request, float], bytes]


class ModelCallError(RuntimeError):
    """A model endpoint could not produce an answer."""


def _urlopen(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class OpenAIChatClient:
    def __init__(self, transport: Transport | None = None, timeout_s: float = 120.0) -> None:
        self._transport = transport or _urlopen
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
    if not endpoint.api_key_env:
        return {}
    key = os.environ.get(endpoint.api_key_env, "").strip()
    if not key:
        raise ModelCallError(f"{endpoint.api_key_env} is not set, and {endpoint.model} needs it as its API key")
    return {"Authorization": f"Bearer {key}"}


def _content(raw: bytes, where: str) -> str:
    try:
        payload: Any = json.loads(raw)
        content = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ModelCallError(f"{where} sent a malformed reply: {raw[:200]!r}") from exc
    if not isinstance(content, str):
        raise ModelCallError(f"{where} sent a malformed reply: content is {type(content).__name__}")
    return content
