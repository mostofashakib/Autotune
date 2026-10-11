"""Automatic dataset format detection, validation, and normalization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd

from autotune.core.domain.models import ChatMessage, CleanedDataset, DatasetRecord, TaskType
from autotune.core.ports.sanitizer import DataSanitizer


class AutoFormatSanitizer(DataSanitizer):
    """Parses arbitrary dataset files into normalized ChatMessage records."""

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
            if "total_reward" in row or "per_step_rewards" in row or "rewards" in row or "reward" in row:
                task_type = TaskType.RL_GRPO
            elif task_type != TaskType.RL_GRPO and any("tool" in m.role for m in normalized.messages):
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
        messages: list[ChatMessage] = []

        # Schema 1: Standard 'messages' array
        if "messages" in row and isinstance(row["messages"], list):
            for m in row["messages"]:
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
