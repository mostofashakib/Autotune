"""DataSanitizer protocol defining the ingestion port."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable
from autotune.core.domain.models import CleanedDataset


@runtime_checkable
class DataSanitizer(Protocol):
    """Port for auto-detecting, cleaning, and validating input datasets."""

    def sanitize(self, file_path: Path, val_split_ratio: float = 0.1) -> CleanedDataset:
        """Read an upload (structured rows, documents, or a zip of both), validate, and return train/val splits."""
        ...


class IngestionError(ValueError):
    """An upload cannot become training data. The message names the file and the fix."""
