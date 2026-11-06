from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from autotune.infrastructure.http import http_request


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self._reply(200, self.headers.get("X-Test", "").encode() + self.rfile.read(length))

    def do_GET(self) -> None:
        if self.path == "/missing":
            self._reply(404, b"missing")
        else:
            self._reply(200, b"ok")

    def _reply(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join()


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
