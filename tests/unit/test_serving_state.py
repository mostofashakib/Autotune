"""vLLM output goes to a log file, and the provider reports the server's real state."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

from autotune.infrastructure.serving.vllm_provider import VLLMServingProvider


def _provider(tmp_path, *, exit_code=None, healthy=False):
    process = MagicMock()
    process.pid = 4242
    process.poll.return_value = exit_code
    runner = MagicMock(return_value=process)
    request = (lambda *a, **k: (200, b"OK")) if healthy else (lambda *a, **k: (_ for _ in ()).throw(ConnectionError("refused")))
    return VLLMServingProvider(process_runner=runner, request_fn=request), runner


def test_vllm_output_is_written_to_a_log_file_instead_of_an_unread_pipe(tmp_path):
    provider, runner = _provider(tmp_path)

    provider.start("Qwen/Qwen2.5-0.5B-Instruct", port=8011, log_path=tmp_path / "vllm.log")

    kwargs = runner.call_args.kwargs
    assert kwargs["stdout"] is not subprocess.PIPE
    assert kwargs["stdout"].name == str(tmp_path / "vllm.log")
    assert kwargs["stderr"] is subprocess.STDOUT


def test_a_provider_that_never_started_is_stopped(tmp_path):
    provider, _ = _provider(tmp_path)

    assert provider.state().status == "stopped"


def test_a_running_server_that_is_not_answering_yet_is_starting(tmp_path):
    provider, _ = _provider(tmp_path)
    provider.start("m", port=8011, log_path=tmp_path / "vllm.log")

    state = provider.state()

    assert (state.status, state.base_url) == ("starting", "http://127.0.0.1:8011")


def test_a_server_that_answers_health_checks_is_online(tmp_path):
    provider, _ = _provider(tmp_path, healthy=True)
    provider.start("m", port=8011, log_path=tmp_path / "vllm.log")

    assert provider.state().status == "online"


def test_a_server_that_exited_has_failed_and_shows_its_log(tmp_path):
    provider, _ = _provider(tmp_path, exit_code=1)
    log = tmp_path / "vllm.log"
    provider.start("m", port=8011, log_path=log)
    log.write_text("loading model\nModuleNotFoundError: No module named 'vllm'\n")

    state = provider.state()

    assert state.status == "failed"
    assert "No module named 'vllm'" in state.log_tail
