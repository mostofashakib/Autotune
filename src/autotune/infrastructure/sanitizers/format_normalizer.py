"""Automatic dataset format detection, validation, and normalization."""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd

from autotune.core.domain.models import ChatMessage, CleanedDataset, DatasetRecord, TaskType
from autotune.core.ports.sanitizer import DataSanitizer, IngestionError
from autotune.infrastructure.sanitizers.text_decoding import decode_text

STRUCTURED_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".pq"}


@dataclass(frozen=True)
class StructuredFile:
    """One structured file: conversation records, or raw text rows that still need generation."""
    task_type: TaskType
    records: list[DatasetRecord]
    texts: list[tuple[str, str]]
    dropped: int
    warnings: list[str]


class AutoFormatSanitizer(DataSanitizer):
    """Normalize one structured file. Raw text rows need DocumentIngestor instead."""

    def sanitize(self, file_path: Path, val_split_ratio: float = 0.1) -> CleanedDataset:
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        parsed = normalize_file(file_path.name, file_path.read_bytes())
        if parsed.texts:
            raise IngestionError(f"{file_path.name} holds raw text, not conversations. Set AUTOTUNE_DOCUMENT_MODEL and "
                                 "upload it so Autotune generates training pairs from it.")
        train_records, val_records = tail_split(parsed.records, val_split_ratio)
        return CleanedDataset(
            task_type=parsed.task_type,
            total_records=len(parsed.records),
            train_records=train_records,
            val_records=val_records,
            dropped_count=parsed.dropped,
            warnings=parsed.warnings,
        )


def normalize_file(name: str, data: bytes) -> StructuredFile:
    """Normalize rows with order-independent task precedence.

    Reward labels outrank preference pairs, then tool use, then chat. Only
    valid retained rows influence classification. Chosen/rejected rewards
    validate preference order and do not by themselves select GRPO. A file
    with no conversation rows but a text column yields its texts instead.
    """
    raw_rows, bad_lines = read_rows(name, data)
    records: list[DatasetRecord] = []
    dropped_count = len(bad_lines)
    warnings: list[str] = []
    if bad_lines:
        shown = ", ".join(map(str, bad_lines[:10])) + (" and more" if len(bad_lines) > 10 else "")
        warnings.append(f"Dropped {len(bad_lines)} malformed JSON lines in {name} (lines {shown}).")

    task_type = TaskType.CHAT
    for idx, row in enumerate(raw_rows):
        normalized = _normalize_row(row, idx)
        if normalized is None:
            continue
        records.append(normalized)
        if any(_reward_present(row.get(key)) for key in ("total_reward", "per_step_rewards", "rewards", "reward")):
            task_type = TaskType.RL_GRPO
        elif task_type != TaskType.RL_GRPO and "chosen" in normalized.metadata and "rejected" in normalized.metadata:
            task_type = TaskType.PREFERENCE
        elif task_type not in (TaskType.RL_GRPO, TaskType.PREFERENCE) and any(m.role == "tool" for m in normalized.messages):
            task_type = TaskType.TOOL_USE

    texts = [] if records else _text_rows(name, raw_rows)
    if not records and not texts:
        raise IngestionError(f"No valid records could be extracted from {name}")
    invalid_rows = len(raw_rows) - len(records or texts)
    if invalid_rows > 0:
        warnings.append(f"Dropped {invalid_rows} invalid or empty rows.")
    return StructuredFile(task_type, records, texts, dropped_count + invalid_rows, warnings)


def tail_split(records: list[DatasetRecord], val_split_ratio: float) -> tuple[list[DatasetRecord], list[DatasetRecord]]:
    """Hold out the last rows for validation, keeping at least one training row."""
    if val_split_ratio > 0.0 and len(records) >= 2:
        val_count = min(len(records) - 1, max(1, int(len(records) * val_split_ratio)))
    else:
        val_count = 0
    return (records[:-val_count], records[-val_count:]) if val_count else (records, [])


_TEXT_KEYS = ("text", "content", "body", "document")


def _text_rows(name: str, rows: list[Any]) -> list[tuple[str, str]]:
    """Raw text from the first text column, keyed by row so splits keep rows apart."""
    texts = []
    for idx, row in enumerate(rows, start=1):
        text = _first_text(row, _TEXT_KEYS) if isinstance(row, dict) else ""
        if text:
            texts.append((f"{name}#row{idx}", text))
    return texts


def _normalize_row(row: dict[str, Any], idx: int) -> DatasetRecord | None:
    if not isinstance(row, dict):
        return None
    # Missing columns in mixed CSV/Parquet rows arrive as NaN.
    row = {k: v for k, v in row.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    if "chosen" in row or "rejected" in row:
        return _preference(row, idx)
    messages = _conversation_messages(row)
    if len(messages) < 2:
        return None

    record_id = str(row.get("id") or row.get("episode_id") or f"rec_{idx:06d}")
    meta = {k: v for k, v in row.items() if k not in _CONVERSATION_KEYS}
    return DatasetRecord(id=record_id, messages=messages, metadata=meta)


_MESSAGE_KEYS = ("messages", "conversations")
_PROMPT_KEYS = ("prompt", "question", "input", "query", "user")
_RESPONSE_KEYS = ("completion", "answer", "output", "response", "assistant")
_CONVERSATION_KEYS = {*_MESSAGE_KEYS, *_PROMPT_KEYS, *_RESPONSE_KEYS, "instruction"}
_ROLES = {"system": "system", "user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant",
          "bot": "assistant", "model": "assistant", "ai": "assistant", "tool": "tool", "function": "tool"}


def _conversation_messages(row: dict[str, Any]) -> list[ChatMessage]:
    """Messages from a message list, an Alpaca row, or any prompt and response column pair."""
    raw = next((row[key] for key in _MESSAGE_KEYS if key in row), None)
    if raw is not None:
        return [m for m in map(_message, _as_list(raw)) if m]
    response = _first_text(row, _RESPONSE_KEYS)
    instruction = _text(row.get("instruction"))
    if instruction and response:
        context = _text(row.get("input"))
        prompt = f"{instruction}\n\nContext: {context}" if context else instruction
    else:
        prompt = _first_text(row, _PROMPT_KEYS)
    if not prompt or not response:
        return []
    return [ChatMessage(role="user", content=prompt), ChatMessage(role="assistant", content=response)]


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def _message(raw: Any) -> ChatMessage | None:
    if not isinstance(raw, dict):
        return None
    role = _ROLES.get(_text(raw.get("role", raw.get("from"))).lower())
    content = _text(raw.get("content", raw.get("value")))
    return ChatMessage(role=role, content=content) if role and content else None


def _text(value: Any) -> str:
    """Plain text from a string, a number, or a list of content parts."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        parts = (part.get("text") for part in value if isinstance(part, dict))
        return "\n".join(p.strip() for p in parts if isinstance(p, str) and p.strip())
    return str(value).strip()


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    return next((text for text in (_text(row.get(key)) for key in keys) if text), "")


def read_rows(name: str, data: bytes) -> tuple[list[Any], list[int]]:
    """Parse structured file bytes into raw rows and the line numbers of malformed JSON lines."""
    suffix = Path(name).suffix.lower()
    if suffix not in STRUCTURED_SUFFIXES:
        raise IngestionError(f"Unsupported dataset format: {suffix}. Supported: {', '.join(sorted(STRUCTURED_SUFFIXES))}")
    if not data.strip():
        raise IngestionError(f"{name} is empty.")
    try:
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(io.BytesIO(data)).to_dict("records"), []
        text = decode_text(data)
        if suffix in (".csv", ".tsv"):
            return pd.read_csv(io.StringIO(text), sep=_delimiter(suffix, text)).to_dict("records"), []
    except Exception as exc:
        raise IngestionError(f"Could not read {name}. The file is corrupt or malformed ({exc}).") from exc
    return _json_rows(text)


def _delimiter(suffix: str, text: str) -> str:
    if suffix == ".tsv":
        return "\t"
    try:
        return csv.Sniffer().sniff(text[:65536], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


_WRAPPER_KEYS = ("data", "rows", "records", "examples", "items", "train")


def _json_rows(text: str) -> tuple[list[Any], list[int]]:
    text = text.strip()
    if not text:
        return [], []
    if text[0] in "[{":
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            document = None
        if isinstance(document, list):
            return document, []
        if isinstance(document, dict):
            return next((document[k] for k in _WRAPPER_KEYS if isinstance(document.get(k), list)), [document]), []
    rows, bad_lines = [], []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad_lines.append(number)
    return rows, bad_lines


def _reward_present(value: Any) -> bool:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return bool(value) and all(_reward_present(item) for item in value)
    try:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def _response(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        answers = [m.get("content") for m in value if isinstance(m, dict) and m.get("role") == "assistant"]
        if answers and all(isinstance(a, str) and a.strip() for a in answers):
            return "\n".join(a.strip() for a in answers)
    return ""


def _first_user_content(messages: Any) -> str | None:
    """The prompt of a Forge row, which carries it as the user turn inside `chosen`."""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if not isinstance(messages, list):
        return None
    return next((m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)


def _preference(row: dict[str, Any], idx: int) -> DatasetRecord | None:
    prompt = row.get("prompt") if "prompt" in row else _first_user_content(row.get("chosen"))
    chosen, rejected = _response(row.get("chosen")), _response(row.get("rejected"))
    if not isinstance(prompt, str) or not prompt.strip() or not chosen or not rejected or chosen == rejected:
        return None
    if "chosen_reward" in row or "rejected_reward" in row:
        if not all(isinstance(row.get(k), (int, float, str)) and _reward_present(row.get(k)) for k in ("chosen_reward", "rejected_reward")):
            return None
        if float(row["chosen_reward"]) <= float(row["rejected_reward"]):
            return None
    return DatasetRecord(
        id=str(row.get("id") or f"rec_{idx:06d}"), prompt=prompt.strip(),
        messages=[ChatMessage(role="user", content=prompt.strip()), ChatMessage(role="assistant", content=chosen)],
        metadata={**{k: v for k, v in row.items() if k != "prompt"}, "chosen": chosen, "rejected": rejected},
    )
