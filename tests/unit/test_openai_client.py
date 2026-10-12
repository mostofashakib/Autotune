"""The model client talks to any OpenAI-compatible server and fails loudly."""
from __future__ import annotations

import json
import urllib.error

import pytest

from autotune.core.domain.models import ModelEndpoint
from autotune.infrastructure.evaluators.openai_client import ModelCallError, OpenAIChatClient

ENDPOINT = ModelEndpoint(base_url="http://localhost:11434/v1/", model="gemma3:1b")


class Recorder:
    def __init__(self, body: bytes):
        self.body = body
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        return self.body


def _reply(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()


def test_it_posts_a_chat_completion_and_returns_the_text():
    transport = Recorder(_reply("4"))
    client = OpenAIChatClient(transport=transport, timeout_s=30)

    answer = client.complete(ENDPOINT, [{"role": "user", "content": "2+2?"}], temperature=0.2, max_tokens=16)

    request, timeout = transport.requests[0]
    assert answer == "4"
    assert request.full_url == "http://localhost:11434/v1/chat/completions"
    assert json.loads(request.data) == {
        "model": "gemma3:1b", "messages": [{"role": "user", "content": "2+2?"}],
        "temperature": 0.2, "max_tokens": 16,
    }
    assert timeout == 30
    assert request.get_header("Authorization") is None


def test_it_sends_the_key_from_the_named_env_var(monkeypatch):
    monkeypatch.setenv("SERVE_KEY", "sk-live")
    transport = Recorder(_reply("ok"))

    OpenAIChatClient(transport=transport).complete(
        ENDPOINT.model_copy(update={"api_key_env": "SERVE_KEY"}), [{"role": "user", "content": "hi"}],
    )

    assert transport.requests[0][0].get_header("Authorization") == "Bearer sk-live"


def test_a_missing_key_env_var_fails_and_names_it(monkeypatch):
    monkeypatch.delenv("SERVE_KEY", raising=False)
    client = OpenAIChatClient(transport=Recorder(_reply("ok")))

    with pytest.raises(ModelCallError, match="SERVE_KEY"):
        client.complete(ENDPOINT.model_copy(update={"api_key_env": "SERVE_KEY"}), [{"role": "user", "content": "hi"}])


def test_an_unreachable_server_fails_and_names_the_endpoint():
    def refuse(request, timeout):
        raise urllib.error.URLError("connection refused")

    with pytest.raises(ModelCallError, match="gemma3:1b at http://localhost:11434/v1/"):
        OpenAIChatClient(transport=refuse).complete(ENDPOINT, [{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"choices": []}', b'{"choices": [{"message": {}}]}'])
def test_a_malformed_reply_fails_instead_of_scoring_empty_text(body):
    with pytest.raises(ModelCallError, match="malformed"):
        OpenAIChatClient(transport=Recorder(body)).complete(ENDPOINT, [{"role": "user", "content": "hi"}])


def test_a_reply_cut_off_before_any_text_fails_and_says_why():
    body = json.dumps({"choices": [{"finish_reason": "length", "message": {"content": "", "reasoning": "thinking..."}}]}).encode()

    with pytest.raises(ModelCallError, match="max_tokens"):
        OpenAIChatClient(transport=Recorder(body)).complete(ENDPOINT, [{"role": "user", "content": "hi"}])


def test_an_empty_answer_that_finished_normally_is_returned_as_is():
    body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}).encode()

    assert OpenAIChatClient(transport=Recorder(body)).complete(ENDPOINT, [{"role": "user", "content": "hi"}]) == ""
