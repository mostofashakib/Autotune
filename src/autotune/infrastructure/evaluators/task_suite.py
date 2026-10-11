"""Task and benchmark suite definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Difficulty = int  # 1–5


@dataclass
class Task:
    """A single benchmark task for policy evaluation."""

    name: str
    domain: str
    objective: str
    success_fn: Callable[[dict[str, Any]], bool]
    difficulty: Difficulty
    template: Any = None
