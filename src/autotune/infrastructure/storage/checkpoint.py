"""Checkpoint management and manifest serialization for trained policies."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

MANIFEST_NAME = "policy_checkpoint.json"


class PolicyCheckpoint(BaseModel):
    """Manifest describing a policy trained from graded rollouts."""
    model_config = ConfigDict(protected_namespaces=())

    objective: str  # "grpo" | "dpo" | "sft"
    training_mode: str = "online"  # "online" | "offline"
    base_model: str
    model_path: str  # what the runtime agent serves (local path or served name)
    num_examples: int
    mean_reward: float
    created_at: str = ""
    experiment_config: dict[str, Any] = Field(default_factory=dict)
    train_envs: list[str] = Field(default_factory=list)
    seed: int | None = None
    run_id: str = ""

    def save(self, out_dir: Path) -> Path:
        """Serialize checkpoint manifest to target directory."""
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        manifest_path = target / MANIFEST_NAME
        manifest_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return manifest_path

    @classmethod
    def load(cls, out_dir: Path) -> "PolicyCheckpoint":
        """Load checkpoint manifest from target directory."""
        manifest_path = Path(out_dir) / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no policy checkpoint manifest at {manifest_path}")
        return cls.model_validate_json(manifest_path.read_text(encoding="utf-8"))
