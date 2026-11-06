"""The one HTTP transport every infrastructure adapter shares."""
from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable, Mapping

# (method, url, headers=..., body=...) -> (status, body)
HttpCall = Callable[..., tuple[int, bytes]]


def http_request(
    method: str, url: str, *, headers: Mapping[str, str], body: bytes | None = None, timeout: float,
) -> tuple[int, bytes]:
    """Send one request. An HTTP error status is returned. A network failure raises ConnectionError."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except OSError as exc:
        raise ConnectionError(f"HTTP request to {url} failed: {exc}") from exc
