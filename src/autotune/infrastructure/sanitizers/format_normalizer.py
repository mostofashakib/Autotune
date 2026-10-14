"""Automatic dataset format detection, validation, and normalization."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import pandas as pd

from autotune.core.domain.models import ChatMessage, CleanedDataset, DatasetRecord, TaskType
from autotune.core.ports.sanitizer import DataSanitizer


class AutoFormatSanitizer(DataSanitizer):
    """Normalize structured rows with order-independent task precedence.

    Reward labels outrank preference pairs, then tool use, then chat. Only
    valid retained rows influence classification. Chosen/rejected rewards
    validate preference order and do not by themselves select GRPO.
    """

    def sanitize(self, file_path: Path, val_split_ratio: float = 0.1) -> CleanedDataset:
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        records: list[DatasetRecord] = []
        dropped_count = 0
        warnings: list[str] = []

        suffix = file_path.suffix.lower()
        if suffix in (".jsonl", ".json"):
            raw_rows = self._read_jsonl(file_path)
        elif suffix in (".parquet", ".pq"):
            df = pd.read_parquet(file_path)
            raw_rows = df.to_dict("records")
        elif suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(file_path, sep=sep)
            raw_rows = df.to_dict("records")
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}. Supported: .jsonl, .json, .csv, .parquet")

        task_type = TaskType.CHAT
        for idx, row in enumerate(raw_rows):
            normalized = self._normalize_row(row, idx)
            if normalized is None:
                dropped_count += 1
                continue

            records.append(normalized)
            if any(_reward_present(row.get(key)) for key in ("total_reward", "per_step_rewards", "rewards", "reward")):
                task_type = TaskType.RL_GRPO
            elif task_type != TaskType.RL_GRPO and "chosen" in normalized.metadata and "rejected" in normalized.metadata:
                task_type = TaskType.PREFERENCE
            elif task_type not in (TaskType.RL_GRPO, TaskType.PREFERENCE) and any(m.role == "tool" for m in normalized.messages):
                task_type = TaskType.TOOL_USE

        if not records:
            raise ValueError(f"No valid records could be extracted from {file_path}")

        # Train/val split
        if val_split_ratio > 0.0 and len(records) >= 2:
            val_count = max(1, int(len(records) * val_split_ratio))
            if val_count >= len(records):
                val_count = len(records) - 1
        else:
            val_count = 0

        train_records = records[:-val_count] if val_count > 0 else records
        val_records = records[-val_count:] if val_count > 0 else []

        if dropped_count > 0:
            warnings.append(f"Dropped {dropped_count} invalid or empty rows.")

        return CleanedDataset(
            task_type=task_type,
            total_records=len(records),
            train_records=train_records,
            val_records=val_records,
            dropped_count=dropped_count,
            warnings=warnings,
        )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        text = path.read_text().strip()
        if not text:
            return []

        # Check if entire file is a single JSON array
        if text.startswith("[") and text.endswith("]"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _normalize_row(self, row: dict[str, Any], idx: int) -> DatasetRecord | None:
        if not isinstance(row, dict):
            return None
        # Missing columns in mixed CSV/Parquet rows arrive as NaN.
        row = {k: v for k, v in row.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
        if "chosen" in row or "rejected" in row:
            return _preference(row, idx)
        messages: list[ChatMessage] = []

        raw_messages = row.get("messages")
        if hasattr(raw_messages, "tolist"):
            raw_messages = raw_messages.tolist()
        if isinstance(raw_messages, str):
            try:
                raw_messages = json.loads(raw_messages)
            except json.JSONDecodeError:
                raw_messages = None

        # Schema 1: Standard 'messages' array
        if isinstance(raw_messages, list):
            for m in raw_messages:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    role = str(m["role"]).lower()
                    if role in ("system", "user", "assistant", "tool") and str(m["content"]).strip():
                        messages.append(ChatMessage(role=role, content=str(m["content"]).strip()))

        # Schema 2: 'prompt' and 'completion'
        elif "prompt" in row and "completion" in row:
            p = str(row["prompt"]).strip()
            c = str(row["completion"]).strip()
            if p and c:
                messages = [
                    ChatMessage(role="user", content=p),
                    ChatMessage(role="assistant", content=c),
                ]

        # Schema 3: 'instruction' and 'output' (Alpaca format)
        elif "instruction" in row and "output" in row:
            inst = str(row["instruction"]).strip()
            inp = str(row.get("input") or "").strip()
            out = str(row["output"]).strip()
            if inst and out:
                user_msg = f"{inst}\n\nContext: {inp}" if inp else inst
                messages = [
                    ChatMessage(role="user", content=user_msg),
                    ChatMessage(role="assistant", content=out),
                ]

        # Schema 4: 'question' and 'answer'
        elif "question" in row and "answer" in row:
            q = str(row["question"]).strip()
            a = str(row["answer"]).strip()
            if q and a:
                messages = [
                    ChatMessage(role="user", content=q),
                    ChatMessage(role="assistant", content=a),
                ]

        if len(messages) < 2:
            return None

        record_id = str(row.get("id") or row.get("episode_id") or f"rec_{idx:06d}")
        meta = {k: v for k, v in row.items() if k not in ("messages", "prompt", "completion", "instruction", "output", "question", "answer")}
        return DatasetRecord(id=record_id, messages=messages, metadata=meta)


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


def _preference(row: dict[str, Any], idx: int) -> DatasetRecord | None:
    prompt = row.get("prompt")
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
