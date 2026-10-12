"""An LLM judge that scores an answer against the reference from 0 to 1."""
from __future__ import annotations

import json
import re
from typing import Protocol

from autotune.core.domain.models import ModelEndpoint

_SYSTEM = (
    "You grade a model's answer against a reference answer. Judge whether the answer "
    "accomplishes what the reference does, allowing different wording. Reply with JSON "
    'only: {"score": <number from 0 to 1>, "reason": "<one sentence>"}.'
)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class JudgeError(RuntimeError):
    """The judge replied, but not with a usable score."""


class ChatClient(Protocol):
    def complete(
        self, endpoint: ModelEndpoint, messages: list[dict[str, str]],
        *, temperature: float = 0.7, max_tokens: int = 1024,
    ) -> str: ...


class LLMJudge:
    def __init__(self, client: ChatClient, endpoint: ModelEndpoint) -> None:
        self._client = client
        self._endpoint = endpoint

    def score(self, prompt: str, reference: str, answer: str) -> float:
        user = f"Task:\n{prompt}\n\nReference answer:\n{reference}\n\nModel answer:\n{answer}"
        reply = self._client.complete(
            self._endpoint, [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=4096,
        )
        return _parse_score(reply)


def _parse_score(reply: str) -> float:
    match = _JSON_OBJECT.search(reply)
    try:
        value = json.loads(match.group(0))["score"] if match else None
    except (ValueError, KeyError, TypeError) as exc:
        raise JudgeError(f"judge reply is not a JSON score: {reply[:200]!r}") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise JudgeError(f"judge score must be a number from 0 to 1: {reply[:200]!r}")
    return float(value)
