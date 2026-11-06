"""The CLI and the API start from the same defaults, taken from autotune.defaults."""
from __future__ import annotations

import click
import typer.main

from autotune import defaults
from autotune.core.domain.models import RunPodConfig
from autotune.presentation.api.routes.benchmarks import BenchmarkRequest
from autotune.presentation.api.routes.cloud import RunPodLaunchRequest
from autotune.presentation.api.routes.serving import ServingRequest
from autotune.presentation.api.routes.training import TrainingStartRequest
from autotune.presentation.cli.main import app


def _cli_default(*command_path: str, option: str) -> object:
    command: click.Command = typer.main.get_command(app)
    for name in command_path:
        command = command.commands[name]  # type: ignore[attr-defined]
    default = next(p.default for p in command.params if p.name == option)
    return str(default) if hasattr(default, "parts") else default


def test_cli_and_api_defaults_match() -> None:
    assert _cli_default("train", option="output_dir") == defaults.CHECKPOINT_DIR
    assert TrainingStartRequest.model_fields["output_dir"].default == defaults.CHECKPOINT_DIR
    assert ServingRequest.model_fields["model_path"].default == defaults.CHECKPOINT_DIR

    assert _cli_default("serve", option="port") == defaults.SERVE_PORT
    assert ServingRequest.model_fields["port"].default == defaults.SERVE_PORT

    for command in (("benchmark", "dataset"), ("benchmark", "harbor")):
        assert _cli_default(*command, option="k") == defaults.PASS_K
        assert _cli_default(*command, option="output") == defaults.BENCHMARK_DIR
    assert BenchmarkRequest.model_fields["k"].default == defaults.PASS_K
    assert BenchmarkRequest.model_fields["output_dir"].default == defaults.BENCHMARK_DIR
    assert _cli_default("benchmark", "dataset", option="max_tokens") == defaults.ANSWER_MAX_TOKENS
    assert BenchmarkRequest.model_fields["max_tokens"].default == defaults.ANSWER_MAX_TOKENS
    assert _cli_default("benchmark", "harbor", option="jobs_dir") == defaults.HARBOR_JOBS_DIR
    assert BenchmarkRequest.model_fields["jobs_dir"].default == defaults.HARBOR_JOBS_DIR

    assert _cli_default("cloud", "launch", option="gpu_type") == defaults.RUNPOD_GPU
    assert RunPodLaunchRequest.model_fields["gpu_type"].default == defaults.RUNPOD_GPU
    assert _cli_default("cloud", "launch", option="pod_name") == defaults.RUNPOD_POD_NAME
    assert RunPodLaunchRequest.model_fields["pod_name"].default == defaults.RUNPOD_POD_NAME
    assert _cli_default("cloud", "launch", option="disk_gb") == defaults.RUNPOD_DISK_GB
    assert RunPodLaunchRequest.model_fields["disk_gb"].default == defaults.RUNPOD_DISK_GB


def test_single_disk_sizes_container_and_volume_alike() -> None:
    config = RunPodConfig.single_disk("NVIDIA A100", "my-pod", 80)
    assert (config.gpu_type_id, config.pod_name) == ("NVIDIA A100", "my-pod")
    assert config.container_disk_in_gb == config.volume_in_gb == 80
