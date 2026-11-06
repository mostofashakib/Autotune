from pathlib import Path
from unittest.mock import MagicMock
from autotune.infrastructure.serving.vllm_provider import VLLMServingProvider


def test_vllm_start_and_stop():
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.wait.return_value = 0
    mock_runner = MagicMock(return_value=mock_process)

    provider = VLLMServingProvider(process_runner=mock_runner)
    endpoint = provider.start(model_path=Path("/tmp/model"), model_name="test-llama", port=8080)

    assert endpoint.model_name == "test-llama"
    assert endpoint.port == 8080
    assert endpoint.base_url == "http://127.0.0.1:8080"
    assert endpoint.backend == "vllm"

    mock_runner.assert_called_once()
    cmd = mock_runner.call_args[0][0]
    assert "--model" in cmd
    assert "/tmp/model" in cmd

    provider.stop()
    mock_process.terminate.assert_called_once()


def test_vllm_is_healthy():
    def mock_http(method: str, url: str, headers: dict, body: bytes | None):
        if url.endswith("/health"):
            return 200, b"OK"
        return 404, b"Not Found"

    provider = VLLMServingProvider(request_fn=mock_http)
    assert provider.is_healthy("http://127.0.0.1:8000") is True

    # When request throws connection error
    def failing_http(method: str, url: str, headers: dict, body: bytes | None):
        raise ConnectionError("Refused")

    unhealthy_provider = VLLMServingProvider(request_fn=failing_http)
    assert unhealthy_provider.is_healthy("http://127.0.0.1:8000") is False
