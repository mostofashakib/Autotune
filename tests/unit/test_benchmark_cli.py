"""`autotune benchmark dataset|harbor` runs the real benchmark and prints what it measured."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from autotune.presentation.cli import main as cli

runner = CliRunner()
ENDPOINTS = ["--base-url", "http://b/v1", "--base-model", "base", "--trained-url", "http://t/v1", "--trained-model", "trained"]


class ModelPerEndpoint:
    """Base always misses, trained always answers with the reference."""

    def complete(self, endpoint, messages, *, temperature=0.7, max_tokens=1024):
        return "pwd" if endpoint.model == "trained" else "no idea"


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "sft_pairs.jsonl"
    path.write_text(json.dumps({"prompt": "Where am I?", "completion": "pwd"}) + "\n")
    return path


def test_dataset_benchmark_writes_and_prints_the_measured_report(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "OpenAIChatClient", lambda: ModelPerEndpoint())

    result = runner.invoke(cli.app, ["benchmark", "dataset", str(_dataset(tmp_path)), *ENDPOINTS, "-k", "2", "--output", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "out" / "benchmark_report.json").read_text())
    assert (report["base"]["pass_at_1"], report["trained"]["pass_at_1"]) == (0.0, 1.0)
    assert report["trained"]["samples"] == 2
    assert "+1.00" in result.output


def test_dataset_benchmark_with_an_unreachable_model_fails_and_writes_nothing(tmp_path, monkeypatch):
    from autotune.infrastructure.evaluators.openai_client import ModelCallError

    class Down:
        def complete(self, endpoint, messages, **kwargs):
            raise ModelCallError(f"{endpoint.model} at {endpoint.base_url} did not answer")

    monkeypatch.setattr(cli, "OpenAIChatClient", lambda: Down())

    result = runner.invoke(cli.app, ["benchmark", "dataset", str(_dataset(tmp_path)), *ENDPOINTS, "--output", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "did not answer" in result.output
    assert not (tmp_path / "out").exists()


def test_dataset_benchmark_requires_both_endpoints(tmp_path):
    result = runner.invoke(cli.app, ["benchmark", "dataset", str(_dataset(tmp_path)), "--base-url", "http://b/v1", "--base-model", "base"])

    assert result.exit_code != 0
    assert "trained" in result.output


def test_harbor_benchmark_runs_both_models_on_the_tasks(tmp_path, monkeypatch):
    task = tmp_path / "tasks" / "hello"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')

    def fake_harbor(command, env):
        model = command[command.index("-m") + 1]
        trial = Path(command[command.index("-o") + 1]) / command[command.index("--job-name") + 1] / "t0"
        trial.mkdir(parents=True)
        reward = 1.0 if model == "openai/trained" else 0.0
        (trial / "result.json").write_text(json.dumps({"task_name": "hello", "verifier_result": {"rewards": {"reward": reward}}}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("autotune.infrastructure.evaluators.harbor_benchmark._run", fake_harbor)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/harbor")

    result = runner.invoke(cli.app, ["benchmark", "harbor", str(tmp_path / "tasks"), *ENDPOINTS, "-k", "1",
                                     "--jobs-dir", str(tmp_path / "jobs"), "--output", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "out" / "benchmark_report.json").read_text())
    assert report["mode"] == "harbor"
    assert report["delta_pass_at_1"] == 1.0


def test_the_old_evaluate_command_is_gone():
    assert runner.invoke(cli.app, ["evaluate", "--help"]).exit_code != 0
