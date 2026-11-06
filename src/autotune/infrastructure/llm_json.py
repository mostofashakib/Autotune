"""Extract JSON from a chat model reply, which may wrap it in a fence or prose."""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from autotune.infrastructure.evaluators.openai_client import ModelCallError

_FENCED_BLOCK = re.compile(r"```[\w-]*\s*\n(.*?)\n?\s*```", re.DOTALL)
_OPENERS = re.compile(r"[{\[]")


class MalformedReply(ModelCallError):
    """The model answered, but not with the JSON shape the prompt asked for."""


def _candidates(text: str) -> Iterator[str]:
    yield text.strip()
    for block in _FENCED_BLOCK.finditer(text):
        yield block.group(1)


def _first_embedded_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for opener in _OPENERS.finditer(text):
        try:
            return decoder.raw_decode(text, opener.start())[0]
        except ValueError:
            continue
    raise MalformedReply("model reply does not contain valid JSON")


def parse_json_reply(text: str) -> Any:
    """Parse the whole reply, else a fenced block, else the first JSON value embedded in prose."""
    if not isinstance(text, str):
        raise MalformedReply("model reply does not contain valid JSON")
    for candidate in _candidates(text):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return _first_embedded_value(text)
