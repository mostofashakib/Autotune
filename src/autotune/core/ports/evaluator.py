"""BenchmarkRunner protocol defining the evaluation port."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from autotune.core.domain.models import ModelEndpoint, SampleResult


@runtime_checkable
class BenchmarkRunner(Protocol):
    """Port for measuring one model: graded samples on a fixed set of tasks."""

    def run(self, model: ModelEndpoint) -> list[SampleResult]:
        """Every graded sample for `model`. Raises when the model cannot be measured."""
        ...
