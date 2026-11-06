"""`autotune train` always trains for real. There is no simulated mode."""
from __future__ import annotations

import json
import zipfile

import pytest
from typer.testing import CliRunner

from autotune.core.domain.models import JobStatus
from autotune.infrastructure.cloud.runpod_provider import RunPodCloudProvider
from autotune.infrastructure.serving.vllm_provider import VLLMServingProvider
from autotune.presentation.cli import main
from autotune.presentation.cli.main import app

GATED_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
OPEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class FakeProbe:
    def available_vram_gb(self) -> float:
        return 48.0


class RecordingBackend:
    jobs: list = []
    error: Exception | None = None

    def train(self, job, dataset):
        if RecordingBackend.error:
            raise RecordingBackend.error
        RecordingBackend.jobs.append(job)
        return job.model_copy(update={"status": JobStatus.COMPLETED, "checkpoint_dir": job.output_dir})


@pytest.fixture
def cli(monkeypatch):
    RecordingBackend.jobs = []
    RecordingBackend.error = None
    monkeypatch.setattr(main, "CudaProbe", FakeProbe)
    monkeypatch.setattr(main, "TrlPeftBackend", RecordingBackend)
    return CliRunner()


@pytest.fixture
def sft_file(tmp_path):
    path = tmp_path / "sft.jsonl"
    path.write_text("".join(json.dumps({"prompt": f"question {i}", "completion": f"answer {i}"}) + "\n" for i in range(100)))
    return path


def test_the_simulated_training_flag_is_gone(tmp_path):
    data = tmp_path / "sft.jsonl"
    data.write_text('{"prompt": "p", "completion": "c"}\n')

    result = CliRunner().invoke(app, ["train", str(data), "--mock"])

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_cli_train_rejects_gated_model_without_token(cli, sft_file, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("autotune.settings._env_file_values", lambda: {})

    result = cli.invoke(main.app, ["train", str(sft_file), "--base-model", GATED_MODEL])

    assert result.exit_code == 1
    assert "HF_TOKEN" in result.output
    assert RecordingBackend.jobs == []


def test_cli_train_applies_intensity(cli, sft_file):
    cli.invoke(main.app, ["train", str(sft_file), "--base-model", OPEN_MODEL, "--intensity", "balanced"])
    cli.invoke(main.app, ["train", str(sft_file), "--base-model", OPEN_MODEL, "--intensity", "quick"])

    balanced, quick = (job.recipe.epochs for job in RecordingBackend.jobs)
    assert balanced >= 2
    assert quick == -(-balanced // 2)


def test_cli_inspect_reads_a_zip_of_csvs(cli, tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in ("a.csv", "b.csv"):
            rows = "".join(f"q{name}{i},a{i}\n" for i in range(60))
            zf.writestr(name, "prompt,completion\n" + rows)

    result = cli.invoke(main.app, ["inspect", str(archive)])

    assert result.exit_code == 0, result.output
    assert "Total Valid Records" in result.output
    assert "120" in result.output


def test_cli_unexpected_error_is_not_swallowed(cli, sft_file):
    RecordingBackend.error = KeyError("boom")

    result = cli.invoke(main.app, ["train", str(sft_file), "--base-model", OPEN_MODEL])

    assert isinstance(result.exception, KeyError)


def test_cli_train_passes_objective_and_quantization_to_the_backend(cli, tmp_path):
    preferences = tmp_path / "prefs.jsonl"
    preferences.write_text("".join(
        json.dumps({"prompt": f"question {i}", "chosen": f"good {i}", "rejected": f"bad {i}"}) + "\n" for i in range(100)))
    result = cli.invoke(main.app, ["train", str(preferences), "--base-model", OPEN_MODEL,
                                   "--objective", "dpo", "--quantization", "lora_16bit"])

    assert result.exit_code == 0, result.output
    recipe = RecordingBackend.jobs[0].recipe
    assert recipe.objective.value == "dpo"
    assert recipe.quantization.value == "lora_16bit"


def test_cli_train_on_raw_text_without_a_document_model_names_the_setting(cli, tmp_path, monkeypatch):
    monkeypatch.delenv("AUTOTUNE_DOCUMENT_MODEL", raising=False)
    monkeypatch.setattr("autotune.settings._env_file_values", lambda: {})
    notes = tmp_path / "notes.txt"
    notes.write_text("Some raw text about a topic.\n" * 20)

    result = cli.invoke(main.app, ["train", str(notes)])

    assert result.exit_code == 1
    assert "AUTOTUNE_DOCUMENT_MODEL" in result.output


def test_cli_list_gpus_prints_a_provider_failure_without_a_traceback(cli, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(
        main, "RunPodCloudProvider",
        lambda api_key=None: RunPodCloudProvider(api_key=api_key, request_fn=lambda *a, **k: (500, b"upstream down")),
    )

    result = cli.invoke(main.app, ["cloud", "list-gpus"])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "upstream down" in result.output


def test_cli_serve_prints_a_spawn_failure(cli, tmp_path, monkeypatch):
    def failing_runner(*args, **kwargs):
        raise OSError("no vllm")

    monkeypatch.setattr(main, "VLLMServingProvider", lambda: VLLMServingProvider(process_runner=failing_runner))

    result = cli.invoke(main.app, ["serve", str(tmp_path)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Failed to start vLLM serving engine" in result.output


def test_cli_inspect_prints_and_writes_environment_quality(cli, tmp_path):
    archive = tmp_path / "bundle.zip"
    steps = [{"step_index": 0, "action": {"endpoint": "/a"}, "reward": 1.0, "terminated": True, "truncated": False}]
    trajectory = json.dumps({"episode_id": "e1", "env": "mail", "task": "t", "seed": 1, "steps": steps})
    sft = "".join(json.dumps({"prompt": f"q{i}", "completion": f"a{i}"}) + "\n" for i in range(12))
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("trajectories.jsonl", trajectory + "\n")
        zf.writestr("sft_pairs.jsonl", sft)

    result = cli.invoke(main.app, ["inspect", str(archive)])

    assert result.exit_code == 0, result.output
    assert "Environment Quality" in result.output
    assert json.loads((tmp_path / "env_quality.json").read_text())[0]["env_name"] == "mail"
    assert (tmp_path / "env_quality.csv").exists()
