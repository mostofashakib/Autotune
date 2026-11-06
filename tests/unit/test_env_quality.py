"""Environment quality computed from Forge export rows."""
from autotune.core.domain.models import RolloutRecord
from autotune.infrastructure.evaluators.env_quality import compute_env_quality


def episode(episode_id, env, endpoints, rewards, **step_extra):
    steps = [
        {"step_index": i, "action": {"endpoint": e} if e else {}, "reward": r, "terminated": False, "truncated": False, **step_extra}
        for i, (e, r) in enumerate(zip(endpoints, rewards, strict=True))
    ]
    return {"episode_id": episode_id, "env": env, "task": "t", "seed": 1, "steps": steps}


def rollout(episode_id, env, reason):
    return RolloutRecord(episode_id=episode_id, env_name=env, termination_reason=reason)


class FixedCoverage:
    env_name = "mail"

    def coverage_score(self, state):
        return 0.5 if state else 0.0


def test_env_quality_from_export_rows():
    trajectories = [
        episode("e1", "mail", ["/a", "/a"], [0.0, 1.0]),
        episode("e2", "mail", ["/b", "/c"], [0.0, 0.0]),
    ]
    rollouts = [rollout("e1", "mail", "success"), rollout("e2", "mail", "dead_end")]
    [metrics] = compute_env_quality(trajectories, rollouts)
    assert metrics.env_name == "mail"
    assert metrics.num_episodes == 2
    assert metrics.num_steps == 4
    assert metrics.reward_density == 0.25
    assert metrics.action_diversity == 0.75
    assert metrics.dead_end_rate == 0.5
    assert metrics.state_coverage_score is None


def test_env_quality_groups_by_environment():
    trajectories = [episode("e1", "mail", ["/a"], [1.0]), episode("e2", "pm", ["/x", "/y"], [0.0, 0.0])]
    rollouts = [rollout("e1", "mail", "success"), rollout("e2", "pm", "dead_end")]
    by_env = {m.env_name: m for m in compute_env_quality(trajectories, rollouts)}
    assert set(by_env) == {"mail", "pm"}
    assert by_env["mail"].reward_density == 1.0
    assert by_env["mail"].dead_end_rate == 0.0
    assert by_env["pm"].action_diversity == 1.0
    assert by_env["pm"].dead_end_rate == 1.0


def test_env_quality_with_manifest_scores_coverage():
    trajectories = [episode("e1", "mail", ["/a", "/b"], [0.0, 0.0], state_after={"k": 1})]
    [metrics] = compute_env_quality(trajectories, [], FixedCoverage())
    assert metrics.state_coverage_score == 0.5


def test_env_quality_without_state_after_leaves_coverage_unscored():
    [metrics] = compute_env_quality([episode("e1", "mail", ["/a"], [0.0])], [], FixedCoverage())
    assert metrics.state_coverage_score is None


def test_env_quality_without_rollouts_reports_zero_dead_ends():
    [metrics] = compute_env_quality([episode("e1", "mail", ["/a"], [1.0])], [])
    assert metrics.dead_end_rate == 0.0
    assert metrics.num_episodes == 1


def test_env_quality_without_trajectories_counts_rollout_episodes():
    [metrics] = compute_env_quality([], [rollout("e1", "mail", "dead_end"), rollout("e2", "mail", "success")])
    assert (metrics.num_episodes, metrics.num_steps) == (2, 0)
    assert (metrics.reward_density, metrics.action_diversity) == (0.0, 0.0)
    assert metrics.dead_end_rate == 0.5


def test_env_quality_ignores_steps_without_an_endpoint():
    [metrics] = compute_env_quality([episode("e1", "mail", ["/a", None, "/a"], [0.0, 0.0, 0.0])], [])
    assert metrics.num_steps == 3
    assert metrics.action_diversity == 0.5


def test_env_quality_with_no_input_is_empty():
    assert compute_env_quality([], []) == []


def test_env_quality_counts_only_numeric_positive_rewards():
    trajectories = [episode("e1", "mail", ["/a", "/b", "/c", "/d", "/e"], [None, "1.0", True, 0.5, -1.0])]
    [metrics] = compute_env_quality(trajectories, [])
    assert metrics.reward_density == 0.2
