"""What one API server owns: its job store, job runners, live training backends, and model server."""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Request

from autotune.core.domain.models import ServingEndpoint, ServingStatus
from autotune.core.ports.hardware import HardwareProbe
from autotune.core.ports.serving import ServingProvider
from autotune.core.ports.trainer import TrainingBackend
from autotune.core.services.training_planner import TrainingPlanner
from autotune.infrastructure.models.hub_metadata import resolve_model
from autotune.infrastructure.storage.job_store import JobStore
from autotune.presentation.api.jobs import JobRunner


@dataclass
class ServingSlot:
    """The one model server this API runs. Starting a new one stops the previous one first."""

    build_provider: Callable[[], ServingProvider]
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _provider: ServingProvider | None = field(default=None, init=False, repr=False)

    def start(self, model_path: str, port: int) -> ServingEndpoint:
        with self._lock:
            self._stop_current()
            provider = self.build_provider()
            endpoint = provider.start(model_path=model_path, port=port)
            self._provider = provider
            return endpoint

    def stop(self) -> None:
        with self._lock:
            self._stop_current()

    def status(self) -> ServingStatus:
        with self._lock:
            if self._provider is None:
                return ServingStatus(status="stopped")
            return self._provider.state()

    def _stop_current(self) -> None:
        if self._provider is not None:
            self._provider.stop()
            self._provider = None


@dataclass
class AppState:
    store: JobStore
    ingestion: JobRunner
    training: JobRunner
    benchmarks: JobRunner
    serving: ServingSlot
    build_training_backend: Callable[[], TrainingBackend]
    build_hardware_probe: Callable[[], HardwareProbe]
    # Live backends by training job id, so a cancel request reaches the trainer.
    backends: dict[str, TrainingBackend] = field(default_factory=dict)

    def planner(self) -> TrainingPlanner:
        """A planner on the GPU the probe reports right now."""
        return TrainingPlanner(self.build_hardware_probe(), resolve_model)


def get_state(request: Request) -> AppState:
    return request.app.state.autotune


State = Annotated[AppState, Depends(get_state)]
