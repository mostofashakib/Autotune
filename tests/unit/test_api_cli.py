"""`autotune api` runs the API the web UI talks to. The old `ui` command is gone."""
from __future__ import annotations

from typer.testing import CliRunner

from autotune.presentation.cli.main import app


def test_the_api_command_runs_the_app_on_the_given_port(monkeypatch):
    import uvicorn

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app_instance, host, port, log_level: seen.update(host=host, port=port))

    result = CliRunner().invoke(app, ["api", "--port", "7861"])

    assert result.exit_code == 0, result.output
    assert seen == {"host": "127.0.0.1", "port": 7861}


def test_the_ui_command_is_gone():
    assert CliRunner().invoke(app, ["ui", "--help"]).exit_code != 0


def test_serve_prints_the_real_api_url(monkeypatch):
    from autotune.core.domain.models import ServingEndpoint
    from autotune.presentation.cli import main
    monkeypatch.setattr(main.VLLMServingProvider, "start", lambda self, **kw: ServingEndpoint(
        model_name="m", base_url="http://127.0.0.1:9001", port=9001, backend="vllm", is_ready=False))

    result = CliRunner().invoke(main.app, ["serve", "./ckpt", "--port", "9001"])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:9001/v1" in result.output
    assert "{port}" not in result.output


def test_export_gguf_reports_a_missing_llama_cpp_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_CPP_DIR", "")

    result = CliRunner().invoke(app, ["export-gguf", str(tmp_path), "--output", str(tmp_path / "m.gguf")])

    assert result.exit_code == 1
    assert "Set LLAMA_CPP_DIR to a llama.cpp checkout" in result.output
