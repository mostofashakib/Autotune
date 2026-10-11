"""BenchmarkEvaluator protocol defining the evaluation port."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable
from autotune.core.domain.models import EvaluationReport


@runtime_checkable
class BenchmarkEvaluator(Protocol):
    """Port for benchmarking trained checkpoints against baseline models."""

    def evaluate(
        self,
        base_model: str,
        checkpoint_dir: Path,
        eval_dataset_path: Path | None = None,
    ) -> EvaluationReport:
        """Run held-out evaluation tasks and return side-by-side comparison report."""
        ...
