"""`autotune train` always trains for real. There is no simulated mode."""
from __future__ import annotations

from typer.testing import CliRunner

from autotune.presentation.cli.main import app


def test_the_simulated_training_flag_is_gone(tmp_path):
    data = tmp_path / "sft.jsonl"
    data.write_text('{"prompt": "p", "completion": "c"}\n')

    result = CliRunner().invoke(app, ["train", str(data), "--mock"])

    assert result.exit_code != 0
    assert "No such option" in result.output
