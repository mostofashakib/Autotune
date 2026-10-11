"""DataSanitizer protocol defining the ingestion port."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable
from autotune.core.domain.models import CleanedDataset


@runtime_checkable
class DataSanitizer(Protocol):
    """Port for auto-detecting, cleaning, and validating input datasets."""

    def sanitize(self, file_path: Path, val_split_ratio: float = 0.1) -> CleanedDataset:
        """Read arbitrary dataset file (CSV, JSONL, Parquet), validate, and return train/val splits."""
        ...
