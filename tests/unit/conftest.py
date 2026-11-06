from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        if self.path.startswith("/unauthorized"):
            self.rfile.read(length)
            self._reply(401, b"bad key: " + b"x" * 300)
            return
        self._reply(200, self.headers.get("X-Test", "").encode() + self.rfile.read(length))

    def do_GET(self) -> None:
        if self.path == "/missing":
            self._reply(404, b"missing")
        elif self.path == "/unauthorized":
            self._reply(401, b"bad key: " + b"x" * 300)
        elif self.path == "/malformed":
            self.wfile.write(b"NOT-HTTP\r\n\r\n")
        elif self.path == "/truncated":
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"short")
            self.close_connection = True
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
    """A real local HTTP server. GET /missing is 404, /unauthorized is 401, /malformed and /truncated break the reply."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join()
