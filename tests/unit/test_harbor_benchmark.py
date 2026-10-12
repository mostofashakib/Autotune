"""Harbor mode: run the model as an agent on Harbor tasks and read each trial's verdict."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autotune.core.domain.models import ModelEndpoint
from autotune.infrastructure.evaluators.harbor_benchmark import HarborBenchmark, HarborError

MODEL = ModelEndpoint(base_url="http://localhost:11434/v1", model="gemma3:1b")


@pytest.fixture
def tasks(tmp_path) -> Path:
    task = tmp_path / "tasks" / "hello"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    return tmp_path / "tasks"


def _trial(task_name, reward=None, exception=None):
    return {
        "task_name": task_name,
        "trial_name": f"{task_name}__x",
        "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
        "exception_info": {"exception_type": "AgentTimeoutError", "exception_message": exception} if exception else None,
    }


class FakeHarbor:
    """Stands in for the harbor process and writes trial results the way Harbor lays them out."""

    def __init__(self, trials, returncode=0):
        self.trials, self.returncode = trials, returncode
        self.calls = []

    def __call__(self, command, env):
        self.calls.append((command, env))
        job_dir = Path(command[command.index("-o") + 1]) / command[command.index("--job-name") + 1]
        job_dir.mkdir(parents=True)
        (job_dir / "result.json").write_text("{}")
        for index, trial in enumerate(self.trials):
            trial_dir = job_dir / f"trial_{index}"
            trial_dir.mkdir()
            (trial_dir / "result.json").write_text(json.dumps(trial))
        return subprocess.CompletedProcess(command, self.returncode, stdout="", stderr="harbor stderr tail")


def _bench(tasks, tmp_path, harbor, **kwargs):
    return HarborBenchmark(tasks, k=2, jobs_dir=tmp_path / "jobs", executable="/usr/bin/harbor", runner=harbor, **kwargs)


def test_it_runs_harbor_with_the_model_endpoint_and_k_attempts(tasks, tmp_path):
    harbor = FakeHarbor([_trial("hello", 1.0), _trial("hello", 0.0)])

    _bench(tasks, tmp_path, harbor).run(MODEL)

    command, env = harbor.calls[0]
    assert command[:2] == ["/usr/bin/harbor", "run"]
    assert command[command.index("-p") + 1] == str(tasks)
    assert command[command.index("-a") + 1] == "terminus-2"
    assert command[command.index("-m") + 1] == "openai/gemma3:1b"
    assert command[command.index("--ak") + 1] == "api_base=http://localhost:11434/v1"
    assert command[command.index("-k") + 1] == "2"
    assert "-y" in command


def test_each_trial_becomes_a_sample_graded_by_its_verifier_reward(tasks, tmp_path):
    harbor = FakeHarbor([_trial("hello", 1.0), _trial("hello", 0.0), _trial("other", 0.5)])

    samples = sorted(_bench(tasks, tmp_path, harbor).run(MODEL), key=lambda s: (s.task_id, s.score))

    assert [(s.task_id, s.passed, s.score, s.error) for s in samples] == [
        ("hello", False, 0.0, None), ("hello", True, 1.0, None), ("other", False, 0.5, None),
    ]


def test_a_crashed_trial_is_a_failed_sample_with_its_error(tasks, tmp_path):
    harbor = FakeHarbor([_trial("hello", exception="agent timed out"), _trial("hello", 1.0)])

    crashed = next(s for s in _bench(tasks, tmp_path, harbor).run(MODEL) if s.error)

    assert not crashed.passed and crashed.score == 0.0
    assert "agent timed out" in crashed.error


def test_a_trial_without_a_verifier_result_is_an_error(tasks, tmp_path):
    samples = _bench(tasks, tmp_path, FakeHarbor([_trial("hello")])).run(MODEL)

    assert samples[0].error == "no verifier result"
    assert not samples[0].passed


def test_a_harbor_run_with_no_trial_results_fails(tasks, tmp_path):
    with pytest.raises(HarborError, match="harbor stderr tail"):
        _bench(tasks, tmp_path, FakeHarbor([], returncode=1)).run(MODEL)


def test_the_api_key_is_passed_through_the_environment_not_the_command(tasks, tmp_path, monkeypatch):
    monkeypatch.setenv("SERVE_KEY", "sk-live")
    harbor = FakeHarbor([_trial("hello", 1.0)])

    _bench(tasks, tmp_path, harbor).run(MODEL.model_copy(update={"api_key_env": "SERVE_KEY"}))

    command, env = harbor.calls[0]
    assert env["OPENAI_API_KEY"] == "sk-live"
    assert "sk-live" not in " ".join(command)


def test_a_missing_key_env_var_fails_before_harbor_runs(tasks, tmp_path, monkeypatch):
    monkeypatch.delenv("SERVE_KEY", raising=False)
    harbor = FakeHarbor([_trial("hello", 1.0)])

    with pytest.raises(HarborError, match="SERVE_KEY"):
        _bench(tasks, tmp_path, harbor).run(MODEL.model_copy(update={"api_key_env": "SERVE_KEY"}))
    assert harbor.calls == []


def test_each_run_gets_its_own_job_directory(tasks, tmp_path):
    bench = _bench(tasks, tmp_path, FakeHarbor([_trial("hello", 1.0)]))
    bench.run(MODEL)
    bench._runner = FakeHarbor([_trial("hello", 0.0)])

    assert [s.passed for s in bench.run(MODEL)] == [False]


def test_a_directory_without_tasks_fails_before_any_run(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no Harbor tasks"):
        HarborBenchmark(tmp_path / "empty", k=1, jobs_dir=tmp_path / "jobs", executable="/usr/bin/harbor")


def test_a_missing_harbor_install_fails_with_a_hint(tasks, tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(HarborError, match="uv tool install harbor"):
        HarborBenchmark(tasks, k=1, jobs_dir=tmp_path / "jobs")


def test_k_must_be_positive(tasks, tmp_path):
    with pytest.raises(ValueError, match="k"):
        HarborBenchmark(tasks, k=0, jobs_dir=tmp_path / "jobs", executable="/usr/bin/harbor")
