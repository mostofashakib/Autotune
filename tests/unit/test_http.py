from __future__ import annotations

import socket

import pytest

from autotune.infrastructure.http import http_request


def test_a_200_returns_status_and_body(server_url):
    assert http_request("GET", server_url + "/", headers={}, timeout=5) == (200, b"ok")


def test_an_error_status_is_returned_not_raised(server_url):
    assert http_request("GET", server_url + "/missing", headers={}, timeout=5) == (404, b"missing")


def test_headers_and_body_reach_the_server(server_url):
    result = http_request("POST", server_url, headers={"X-Test": "h-"}, body=b"payload", timeout=5)
    assert result == (200, b"h-payload")


def test_a_closed_port_raises_connection_error_naming_the_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/"
    with pytest.raises(ConnectionError, match=url):
        http_request("GET", url, headers={}, timeout=5)


@pytest.mark.parametrize("path", ["/malformed", "/truncated"])
def test_a_broken_reply_raises_connection_error(server_url, path):
    with pytest.raises(ConnectionError, match=path):
        http_request("GET", server_url + path, headers={}, timeout=5)
