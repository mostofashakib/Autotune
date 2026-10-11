"""CloudComputeProvider protocol for on-demand cloud GPUs like RunPod."""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from autotune.core.domain.models import RunPodConfig, RunPodInstance


@runtime_checkable
class CloudComputeProvider(Protocol):
    """Port for cloud GPU orchestration (RunPod, Lambda, Modal)."""

    def create_instance(self, config: RunPodConfig) -> RunPodInstance:
        """Provision a remote cloud GPU pod."""
        ...

    def get_instance_status(self, pod_id: str) -> RunPodInstance:
        """Fetch current status and connection details for a pod."""
        ...

    def stop_instance(self, pod_id: str) -> bool:
        """Stop and deallocate cloud GPU pod."""
        ...

    def list_available_gpus(self) -> list[dict[str, str | float]]:
        """List available GPU types and hourly pricing."""
        ...
