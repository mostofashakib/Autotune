"""Extract JSON from a chat model reply, which may wrap it in a fence or prose."""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from autotune.infrastructure.evaluators.openai_client import ModelCallError

_FENCED_BLOCK = re.compile(r"```[\w-]*\s*\n(.*?)\n?\s*```", re.DOTALL)
_SPANS = (re.compile(r"\{.*\}", re.DOTALL), re.compile(r"\[.*\]", re.DOTALL))


class MalformedReply(ModelCallError):
    """The model answered, but not with the JSON shape the prompt asked for."""


def _candidates(text: str) -> Iterator[str]:
    yield text.strip()
    for block in _FENCED_BLOCK.finditer(text):
        yield block.group(1)
    spans = [match for match in (pattern.search(text) for pattern in _SPANS) if match]
    for span in sorted(spans, key=lambda match: match.start()):
        yield span.group(0)


def parse_json_reply(text: str) -> Any:
    """Parse the whole reply, else a fenced block, else the outermost object or array span."""
    if isinstance(text, str):
        for candidate in _candidates(text):
            try:
                return json.loads(candidate)
            except ValueError:
                continue
    raise MalformedReply("model reply does not contain valid JSON")
