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
