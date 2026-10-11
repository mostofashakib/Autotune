import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
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


def test_vllm_generate_success():
    def mock_http(method: str, url: str, headers: dict, body: bytes | None):
        response_payload = {
            "choices": [{"text": " 42 is the answer."}]
        }
        return 200, json.dumps(response_payload).encode("utf-8")

    provider = VLLMServingProvider(request_fn=mock_http)
    result = provider.generate("What is 6 * 7?", "http://127.0.0.1:8000")
    assert "42 is the answer" in result


def test_vllm_generate_error_handling():
    def failing_http(method: str, url: str, headers: dict, body: bytes | None):
        return 500, b"Internal vLLM Error"

    provider = VLLMServingProvider(request_fn=failing_http)
    with pytest.raises(RuntimeError, match="vLLM generate failed with status 500"):
        provider.generate("Hello", "http://127.0.0.1:8000")
