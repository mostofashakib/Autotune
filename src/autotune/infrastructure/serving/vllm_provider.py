"""vLLM Serving Provider adapter implementation."""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from autotune.core.domain.models import ServingEndpoint
from autotune.core.ports.serving import ServingProvider

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
        self._request_fn = request_fn or self._default_http_request
        self._process_runner = process_runner or subprocess.Popen
        self._active_process: subprocess.Popen[Any] | None = None
        self._current_endpoint: ServingEndpoint | None = None

    def start(
        self,
        model_path: Path | str,
        model_name: str = "",
        port: int = 8000,
        gpu_memory_utilization: float = 0.85,
    ) -> ServingEndpoint:
        """
        Launch vLLM server daemon process.

        Args:
            model_path: Local path or HuggingFace repo identifier.
            model_name: Alias name to serve the model under.
            port: HTTP port to listen on.
            gpu_memory_utilization: Fraction of GPU VRAM reserved for vLLM KV-cache.

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

        logger.info("Starting vLLM server: %s", " ".join(cmd))
        try:
            self._active_process = self._process_runner(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        except Exception as e:
            logger.error("Failed to start vLLM process: %s", e)
            raise RuntimeError(f"Could not spawn vLLM server: {e}") from e

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
            if self._current_endpoint:
                self._current_endpoint.is_ready = False

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

    def generate(
        self,
        prompt: str,
        base_url: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate completion via OpenAI-compatible completions API endpoint.

        Args:
            prompt: Text prompt to complete.
            base_url: Base endpoint URL.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated response string.
        """
        api_url = f"{base_url.rstrip('/')}/v1/completions"
        payload = {
            "model": self._current_endpoint.model_name if self._current_endpoint else "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        body_bytes = json.dumps(payload).encode("utf-8")

        status_code, resp_bytes = self._request_fn("POST", api_url, headers=headers, body=body_bytes)
        if status_code != 200:
            raise RuntimeError(f"vLLM generate failed with status {status_code}: {resp_bytes.decode('utf-8', errors='ignore')}")

        parsed = json.loads(resp_bytes.decode("utf-8"))
        choices = parsed.get("choices", [])
        if choices and "text" in choices[0]:
            return choices[0]["text"]
        return ""

    @staticmethod
    def _default_http_request(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float = 10.0,
    ) -> tuple[int, bytes]:
        """Built-in HTTP requester using standard library urllib."""
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:
            raise ConnectionError(f"HTTP request to {url} failed: {e}") from e
