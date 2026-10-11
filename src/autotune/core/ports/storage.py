"""CheckpointStorage protocol defining the persistence port."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class CheckpointStorage(Protocol):
    """Port for persisting and packaging model checkpoints."""

    def save(self, job_id: str, source_dir: Path, target_dir: Path) -> Path:
        """Persist checkpoint directory and return final storage path."""
        ...

    def export_gguf(self, checkpoint_dir: Path, output_file: Path) -> Path:
        """Export adapter or merged model to GGUF format."""
        ...
