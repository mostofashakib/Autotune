"""TrainingBackend protocol defining the execution port."""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from autotune.core.domain.models import CheckpointResult, CleanedDataset, TrainingJob


@runtime_checkable
class TrainingBackend(Protocol):
    """Port for training implementations (TRL, PEFT, Torchtune, Mock)."""

    def train(self, job: TrainingJob, dataset: CleanedDataset) -> CheckpointResult:
        """Execute training and return checkpoint metadata."""
        ...

    def cancel(self, job_id: str) -> None:
        """Cancel an in-flight training run."""
        ...
