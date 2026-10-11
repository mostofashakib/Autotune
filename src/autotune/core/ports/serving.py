"""ServingProvider protocol for inference engines like vLLM."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from autotune.core.domain.models import ServingEndpoint


@runtime_checkable
class ServingProvider(Protocol):
    """Port for model serving and fast inference (vLLM, TGI, Ollama)."""

    def start(self, model_path: Path | str, model_name: str = "", port: int = 8000, gpu_memory_utilization: float = 0.85) -> ServingEndpoint:
        """Start the serving backend daemon and return endpoint details."""
        ...

    def stop(self) -> None:
        """Terminate the running serving process."""
        ...

    def is_healthy(self, base_url: str) -> bool:
        """Check if server is responding to health checks."""
        ...

    def generate(self, prompt: str, base_url: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Send completion request to served model."""
        ...
