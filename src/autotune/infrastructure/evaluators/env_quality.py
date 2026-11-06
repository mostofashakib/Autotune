"""Environment quality diagnostics from a Forge export (trajectories and rollouts)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol

from autotune.core.domain.models import EnvQualityMetrics, RolloutRecord

UNKNOWN_ENV = "unknown"


class StateSchemaCoverageProvider(Protocol):
    """Protocol for state coverage calculation."""

    env_name: str

    def coverage_score(self, state: dict[str, Any]) -> float:
        ...


def compute_env_quality(
    trajectories: list[dict[str, Any]],
    rollouts: list[RolloutRecord],
    manifest: StateSchemaCoverageProvider | None = None,
) -> list[EnvQualityMetrics]:
    """One entry per environment, sorted by name.

    Steps come from `trajectories` (reward density, action diversity, step count) and
    dead ends from `rollouts`. A metric whose source is missing for an environment is 0.0.
    State coverage needs a manifest and per-step `state_after`, otherwise it is None.
    """
    episodes_by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in trajectories:
        episodes_by_env[str(episode.get("env") or UNKNOWN_ENV)].append(episode)
    rollouts_by_env: dict[str, list[RolloutRecord]] = defaultdict(list)
    for rollout in rollouts:
        rollouts_by_env[rollout.env_name or UNKNOWN_ENV].append(rollout)
    return [
        _metrics_for(name, episodes_by_env[name], rollouts_by_env[name], manifest)
        for name in sorted(episodes_by_env.keys() | rollouts_by_env.keys())
    ]


def _metrics_for(
    env_name: str,
    episodes: list[dict[str, Any]],
    rollouts: list[RolloutRecord],
    manifest: StateSchemaCoverageProvider | None,
) -> EnvQualityMetrics:
    steps = [step for episode in episodes for step in episode.get("steps", [])]
    endpoints = [endpoint for step in steps if (endpoint := _endpoint(step))]
    rewarded = sum(1 for step in steps if step.get("reward", 0.0) > 0)
    dead_ends = sum(1 for rollout in rollouts if rollout.termination_reason == "dead_end")
    return EnvQualityMetrics(
        env_name=env_name,
        state_coverage_score=_coverage(steps, manifest),
        reward_density=rewarded / len(steps) if steps else 0.0,
        dead_end_rate=dead_ends / len(rollouts) if rollouts else 0.0,
        action_diversity=len(set(endpoints)) / len(endpoints) if endpoints else 0.0,
        num_episodes=_episode_count(episodes, rollouts),
        num_steps=len(steps),
    )


def _endpoint(step: dict[str, Any]) -> str | None:
    action = step.get("action")
    return action.get("endpoint") if isinstance(action, dict) else None


def _coverage(steps: list[dict[str, Any]], manifest: StateSchemaCoverageProvider | None) -> float | None:
    states = [step["state_after"] for step in steps if isinstance(step.get("state_after"), dict)]
    if manifest is None or not states:
        return None
    return sum(manifest.coverage_score(state) for state in states) / len(states)


def _episode_count(episodes: list[dict[str, Any]], rollouts: list[RolloutRecord]) -> int:
    """Distinct episodes from trajectories, or from rollouts when the environment has no trajectories."""
    if episodes:
        return len({episode.get("episode_id", index) for index, episode in enumerate(episodes)})
    return len({rollout.episode_id for rollout in rollouts})
