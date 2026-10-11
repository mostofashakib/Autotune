"""Map verifiers' graded rewards onto mathematical training signals for GRPO, DPO, and SFT."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev

from autotune.core.domain.models import PreferenceRecord, RolloutRecord

_EPS = 1e-6


@dataclass
class GRPOExample:
    prompt: str
    completion: str
    reward: float
    advantage: float
    behavior_model: str = ""


@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


@dataclass
class SFTExample:
    prompt: str
    completion: str
    behavior_model: str = ""


def grpo_advantages(rollouts: list[RolloutRecord], eps: float = _EPS) -> list[GRPOExample]:
    """Group-relative advantages, one example per rollout in a signal-bearing group."""
    groups: dict[str, list[RolloutRecord]] = {}
    for r in rollouts:
        groups.setdefault(r.prompt, []).append(r)

    examples: list[GRPOExample] = []
    for prompt, group in groups.items():
        if len(group) < 2:
            continue  # no group to be relative to
        rewards = [r.total_reward for r in group]
        spread = pstdev(rewards)
        if spread <= 0:
            continue  # every rollout scored the same — nothing to learn
        mean = sum(rewards) / len(rewards)
        for r in group:
            examples.append(
                GRPOExample(
                    prompt=prompt,
                    completion=r.completion,
                    reward=r.total_reward,
                    advantage=(r.total_reward - mean) / (spread + eps),
                    behavior_model=r.behavior_model,
                )
            )
    return examples


def dpo_examples(preferences: list[PreferenceRecord]) -> list[DPOExample]:
    """Preference examples where the chosen trajectory was graded strictly higher."""
    return [
        DPOExample(prompt=p.prompt, chosen=p.chosen, rejected=p.rejected)
        for p in preferences
        if p.chosen_reward > p.rejected_reward
    ]


def sft_examples(records: list) -> list[SFTExample]:
    """SFT examples from demonstrations with non-empty prompt and completion."""
    return [
        SFTExample(
            prompt=r.prompt,
            completion=r.completion,
            behavior_model=getattr(r, "behavior_model", ""),
        )
        for r in records
        if r.prompt and r.completion
    ]
