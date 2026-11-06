"""`autotune train` always trains for real. There is no simulated mode."""
from __future__ import annotations

import json
import zipfile

import pytest
from typer.testing import CliRunner

from autotune.core.domain.models import JobStatus
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
