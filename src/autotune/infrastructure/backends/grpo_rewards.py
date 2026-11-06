"""Reward function and prompt rows that let TRL's GRPOTrainer train on reward-labeled data."""
from __future__ import annotations

from typing import Any

from autotune.core.domain.models import DatasetRecord
from autotune.core.domain.rewards import row_reward
from autotune.infrastructure.evaluators.scoring import token_f1


def reference_reward(completions: list[Any], reference: list[str], **kwargs: Any) -> list[float]:
    """TRL reward function: token F1 of each completion against its prompt's reference."""
    return [token_f1(_completion_text(completion), ref) for completion, ref in zip(completions, reference, strict=True)]


def grpo_rows(records: list[DatasetRecord]) -> list[dict[str, Any]]:
    """One row per distinct prompt, whose reference is the highest-reward completion (first row wins ties)."""
    best: dict[str, tuple[float, str]] = {}
    for record in records:
        reward = _record_reward(record)
        prompt = _first_content(record, "user")
        if prompt not in best or reward > best[prompt][0]:
            best[prompt] = (reward, _last_content(record, "assistant"))
    return [{"prompt": [{"role": "user", "content": prompt}], "reference": reference}
            for prompt, (_, reference) in best.items()]


def _record_reward(record: DatasetRecord) -> float:
    reward = row_reward(record.metadata)
    if reward is None:
        raise ValueError("GRPO needs a numeric reward (total_reward, reward, rewards, or per_step_rewards) in every row. "
                         "Upload reward-labeled rows or select SFT.")
    return reward


def _first_content(record: DatasetRecord, role: str) -> str:
    return next((m.content for m in record.messages if m.role == role), "")


def _last_content(record: DatasetRecord, role: str) -> str:
    return next((m.content for m in reversed(record.messages) if m.role == role), "")


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    return next((str(m.get("content", "")) for m in reversed(completion)
                 if isinstance(m, dict) and m.get("role") == "assistant"), "")
