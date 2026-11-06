"""The one reader for a row's numeric reward, shared by ingestion, validation, and GRPO."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

SCALAR_REWARD_KEYS = ("total_reward", "reward")
LIST_REWARD_KEYS = ("rewards", "per_step_rewards")


def row_reward(row: Mapping[str, Any]) -> float | None:
    """The first usable reward: total_reward, reward, then the sum of rewards or per_step_rewards."""
    for key in (*SCALAR_REWARD_KEYS, *LIST_REWARD_KEYS):
        reward = reward_value(row.get(key))
        if reward is not None:
            return reward
    return None


def reward_value(value: Any) -> float | None:
    """A finite number, a numeric string, or the sum of a non-empty list of them. Bools and NaN are not rewards."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        items = [_number(item) for item in value]
        if not items or any(item is None for item in items):
            return None
        return float(sum(items))
    return _number(value)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
