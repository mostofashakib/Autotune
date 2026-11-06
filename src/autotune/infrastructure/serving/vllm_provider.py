"""vLLM Serving Provider adapter implementation."""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Callable

from autotune.core.domain.errors import ProviderError
from autotune.core.domain.models import ServingEndpoint, ServingStatus
from autotune.core.ports.serving import ServingProvider
from autotune.infrastructure.http import http_request

logger = logging.getLogger(__name__)


class VLLMServingProvider(ServingProvider):
    """High-throughput inference adapter wrapping vLLM OpenAI-compatible server."""

    def __init__(
        self,
        request_fn: Callable[..., Any] | None = None,
        process_runner: Callable[..., Any] | None = None,
    ) -> None:
        """
        Initialize vLLM provider.

        Args:
            request_fn: Optional dependency-injected HTTP client function for testability.
            process_runner: Optional dependency-injected subprocess runner for testability.
        """
        self._request_fn = request_fn or partial(http_request, timeout=10.0)
        self._process_runner = process_runner or subprocess.Popen
        self._active_process: subprocess.Popen[Any] | None = None
        self._current_endpoint: ServingEndpoint | None = None
        self._log_path: Path | None = None
        self._log_file: Any = None

    def start(
        self,
        model_path: Path | str,
        model_name: str = "",
        port: int = 8000,
        gpu_memory_utilization: float = 0.85,
        log_path: Path | None = None,
    ) -> ServingEndpoint:
        """
        Launch vLLM server daemon process.

        Args:
            model_path: Local path or HuggingFace repo identifier.
            model_name: Alias name to serve the model under.
            port: HTTP port to listen on.
            gpu_memory_utilization: Fraction of GPU VRAM reserved for vLLM KV-cache.
            log_path: File that receives vLLM's output. Defaults to a per-port file in the temp dir.

        Returns:
            ServingEndpoint with base URL and connection status.
        """
        resolved_name = model_name or Path(str(model_path)).name or "autotune-model"
        base_url = f"http://127.0.0.1:{port}"

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_path),
            "--served-model-name",
            resolved_name,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
        ]

        # vLLM logs heavily. An unread pipe would fill and stall the server, so output goes to a file.
        self._log_path = Path(log_path) if log_path else Path(tempfile.gettempdir()) / f"autotune-vllm-{port}.log"
        self._log_file = self._log_path.open("w", encoding="utf-8")
        logger.info("Starting vLLM server: %s (log: %s)", " ".join(cmd), self._log_path)
        try:
            self._active_process = self._process_runner(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        except Exception as e:
            self._log_file.close()
            logger.error("Failed to start vLLM process: %s", e)
            raise ProviderError(f"Could not spawn vLLM server: {e}") from e

        endpoint = ServingEndpoint(
            model_name=resolved_name,
            base_url=base_url,
            port=port,
            backend="vllm",
            is_ready=False,
        )
        self._current_endpoint = endpoint
        return endpoint

    def stop(self) -> None:
        """Terminate the running vLLM process cleanly."""
        if self._active_process is None:
            return

        logger.info("Stopping vLLM serving process (PID: %s)...", self._active_process.pid)
        try:
            self._active_process.terminate()
            try:
                self._active_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._active_process.kill()
                self._active_process.wait(timeout=2)
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.warning("Error stopping vLLM process: %s", e)
        finally:
            self._active_process = None
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            if self._current_endpoint:
                self._current_endpoint.is_ready = False

    def state(self) -> ServingStatus:
        """Observed state: no process, exited, answering health checks, or still starting."""
        if self._active_process is None or self._current_endpoint is None:
            return ServingStatus(status="stopped")
        base_url = self._current_endpoint.base_url
        if self._active_process.poll() is not None:
            return ServingStatus(status="failed", base_url=base_url, log_tail=self._log_tail())
        if self.is_healthy(base_url):
            self._current_endpoint.is_ready = True
            return ServingStatus(status="online", base_url=base_url)
        return ServingStatus(status="starting", base_url=base_url)

    def _log_tail(self, lines: int = 40) -> str:
        if self._log_path is None or not self._log_path.is_file():
            return ""
        return "\n".join(self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    def is_healthy(self, base_url: str) -> bool:
        """
        Check if vLLM instance is ready to receive requests.

        Args:
            base_url: Base endpoint URL, e.g. 'http://127.0.0.1:8000'.

        Returns:
            True if healthcheck succeeds, False otherwise.
        """
        health_url = f"{base_url.rstrip('/')}/health"
        try:
            status_code, _ = self._request_fn("GET", health_url, headers={}, body=None)
            return status_code == 200
        except Exception:
            return False
