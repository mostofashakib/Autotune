"""The judge returns a 0-1 score or fails. It never guesses one."""
from __future__ import annotations

import pytest

from autotune.core.domain.models import ModelEndpoint
from autotune.infrastructure.evaluators.judge import JudgeError, LLMJudge
from autotune.infrastructure.evaluators.openai_client import ModelCallError

JUDGE = ModelEndpoint(base_url="http://judge/v1", model="judge-model")


class FakeClient:
    def __init__(self, reply: str | Exception):
        self.reply = reply
        self.calls = []

    def complete(self, endpoint, messages, *, temperature=0.7, max_tokens=1024):
        self.calls.append((endpoint, messages, temperature))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_it_asks_the_judge_deterministically_and_returns_its_score():
    client = FakeClient('{"score": 0.75, "reason": "mostly right"}')

    score = LLMJudge(client, JUDGE).score("List files", "ls -la", "ls")

    endpoint, messages, temperature = client.calls[0]
    assert score == 0.75
    assert endpoint == JUDGE and temperature == 0.0
    text = " ".join(m["content"] for m in messages)
    assert "List files" in text and "ls -la" in text


def test_a_fenced_json_reply_is_read():
    assert LLMJudge(FakeClient('```json\n{"score": 1}\n```'), JUDGE).score("p", "r", "a") == 1.0


@pytest.mark.parametrize("reply", ["great answer", '{"score": "high"}', '{"score": 1.5}', '{"score": -0.1}', '{"verdict": 1}', '{"score": true}'])
def test_an_unusable_verdict_fails(reply):
    with pytest.raises(JudgeError):
        LLMJudge(FakeClient(reply), JUDGE).score("p", "r", "a")


def test_a_judge_that_cannot_be_reached_fails():
    with pytest.raises(ModelCallError):
        LLMJudge(FakeClient(ModelCallError("down")), JUDGE).score("p", "r", "a")
