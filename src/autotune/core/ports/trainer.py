"""TrainingBackend protocol defining the execution port."""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from autotune.core.domain.models import CheckpointResult, CleanedDataset, TrainingJob, TrainingProgress

ProgressCallback = Callable[[TrainingProgress], None]


@runtime_checkable
class TrainingBackend(Protocol):
    """Port for training implementations (TRL, PEFT, Torchtune)."""

    def train(
        self, job: TrainingJob, dataset: CleanedDataset, on_progress: ProgressCallback | None = None,
    ) -> CheckpointResult:
        """Execute training, reporting each logged step to `on_progress`, and return checkpoint metadata."""
        ...

    def cancel(self, job_id: str) -> None:
        """Cancel an in-flight training run."""
        ...
