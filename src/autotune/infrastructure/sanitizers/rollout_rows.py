"""Turn `grpo_rollouts.parquet` rows into RolloutRecords. A leaf module: it imports nothing from sources."""
from __future__ import annotations

import json
from typing import Any

from autotune.core.domain.errors import MissingExtraError
from autotune.core.domain.models import RolloutRecord

LOAD_COLUMNS = frozenset({"episode_id", "task_name", "prompt", "completion", "total_reward", "passed"})
QUALITY_COLUMNS = frozenset({"episode_id", "env_name", "termination_reason"})


class MalformedExportError(ValueError):
    """An export file exists but does not match the expected schema."""


def read_rollouts(source: Any, label: str, required: frozenset[str]) -> list[RolloutRecord]:
    """Read a parquet path or file object. `required` names the columns the caller cannot do without."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingExtraError("loading rollouts requires pandas") from exc

    try:
        frame = pd.read_parquet(source)
    except Exception as exc:
        raise MalformedExportError(f"could not read rollouts parquet {label}: {exc}") from exc

    missing = required - set(frame.columns)
    if missing:
        raise MalformedExportError(f"rollouts {label} missing columns: {sorted(missing)}")
    return [_rollout_of(row) for row in frame.to_dict("records")]


def _rollout_of(row: dict[str, Any]) -> RolloutRecord:
    prompt = _text(row.get("prompt"))
    return RolloutRecord(
        episode_id=_text(row.get("episode_id")),
        env_name=_text(row.get("env_name")) or environment_of(prompt),
        task_name=_text(row.get("task_name")),
        prompt=prompt,
        completion=_text(row.get("completion")),
        total_reward=_number(row.get("total_reward")),
        passed=bool(row.get("passed")) if not _missing(row.get("passed")) else False,
        per_step_rewards=_parse_per_step(row.get("per_step_rewards")),
        behavior_model=_text(row.get("behavior_model")),
        termination_reason=_text(row.get("termination_reason")) or "unknown",
        verification_results=_parse_json_list(row.get("verification_results")),
        reward_breakdown=_parse_json_dict(row.get("reward_breakdown")),
    )


def environment_of(prompt: str) -> str:
    """Read the environment marker if present."""
    for line in prompt.splitlines():
        if line.startswith("Environment:"):
            return line.partition(":")[2].strip()
    return ""


def _missing(value: Any) -> bool:
    """None or NaN. Arrays and lists are never missing."""
    if value is None:
        return True
    return isinstance(value, float) and value != value


def _text(value: Any) -> str:
    return "" if _missing(value) else str(value)


def _number(value: Any) -> float:
    return 0.0 if _missing(value) else float(value)


def _parse_per_step(value: Any) -> list[float]:
    if _missing(value):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return []


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    parsed = _parse_json(value)
    return parsed if isinstance(parsed, list) else []


def _parse_json_dict(value: Any) -> dict[str, Any]:
    parsed = _parse_json(value)
    return parsed if isinstance(parsed, dict) else {}


def _parse_json(value: Any) -> Any:
    if _missing(value):
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value
