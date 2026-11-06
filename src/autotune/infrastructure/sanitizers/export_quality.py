"""Environment quality from the Forge export files inside an upload."""
from __future__ import annotations

import io
import json
from pathlib import PurePosixPath
from typing import Any

from autotune.core.domain.models import EnvQualityMetrics, RolloutRecord
from autotune.infrastructure.evaluators.env_quality import compute_env_quality
from autotune.infrastructure.sanitizers.rollout_rows import QUALITY_COLUMNS, read_rollouts

TRAJECTORIES_FILE = "trajectories.jsonl"
ROLLOUTS_FILE = "grpo_rollouts.parquet"
# Which Forge export trains first when several are present.
FORGE_TRAINING_PRIORITY = (
    "sft_pairs.jsonl", "sft_dataset.jsonl", "sft_data.jsonl", "batch_export.json", "tasks.json",
    "preference_pairs.jsonl", ROLLOUTS_FILE,
)
FORGE_FILES = frozenset({*FORGE_TRAINING_PRIORITY, TRAJECTORIES_FILE})


def is_trajectories(name: str) -> bool:
    return PurePosixPath(name).name.lower() == TRAJECTORIES_FILE


def is_rollouts(name: str) -> bool:
    return PurePosixPath(name).name.lower() == ROLLOUTS_FILE


def environment_quality(files: list[tuple[str, bytes]]) -> tuple[list[EnvQualityMetrics], list[str]]:
    """Metrics per environment from any trajectories and rollouts files, plus warnings for what was skipped."""
    trajectories: list[dict[str, Any]] = []
    rollouts: list[RolloutRecord] = []
    warnings: list[str] = []
    for name, data in files:
        if is_trajectories(name):
            episodes, skipped = _read_trajectories(data)
            trajectories += episodes
            if skipped:
                warnings.append(f"Ignored {skipped} malformed lines in {name} for environment quality.")
        elif is_rollouts(name):
            try:
                rollouts += _read_rollouts(data)
            except (ValueError, OSError) as exc:
                warnings.append(f"Could not read {name} for environment quality ({exc}).")
    return compute_env_quality(trajectories, rollouts), warnings


def _read_trajectories(data: bytes) -> tuple[list[dict[str, Any]], int]:
    episodes: list[dict[str, Any]] = []
    skipped = 0
    for line in data.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            episode = json.loads(line)
        except json.JSONDecodeError:
            episode = None
        if isinstance(episode, dict) and isinstance(episode.get("steps"), list):
            episodes.append(episode)
        else:
            skipped += 1
    return episodes, skipped


def _read_rollouts(data: bytes) -> list[RolloutRecord]:
    return read_rollouts(io.BytesIO(data), ROLLOUTS_FILE, QUALITY_COLUMNS)
