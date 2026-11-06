"""Typed loaders for dataset files (rollouts, preference pairs, and SFT demonstrations)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autotune.core.domain.models import DatasetRecord, PreferenceRecord, RolloutRecord, SFTRecord
from autotune.core.ports.sanitizer import IngestionError
from autotune.infrastructure.sanitizers.sources import collect_sources


class MalformedExportError(ValueError):
    """An export file exists but does not match the expected schema."""


_ROLLOUT_COLUMNS = {"episode_id", "task_name", "prompt", "completion", "total_reward", "passed"}


def load_rollouts(path: Path) -> list[RolloutRecord]:
    """Load graded episodes from a `grpo_rollouts.parquet` file."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("loading rollouts requires pandas") from exc

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise MalformedExportError(f"could not read rollouts parquet {path}: {exc}") from exc

    missing = _ROLLOUT_COLUMNS - set(df.columns)
    if missing:
        raise MalformedExportError(f"rollouts {path} missing columns: {sorted(missing)}")

    records: list[RolloutRecord] = []
    for row in df.to_dict("records"):
        records.append(
            RolloutRecord(
                episode_id=str(row["episode_id"]),
                env_name=str(row.get("env_name") or _environment_of(str(row["prompt"]))),
                task_name=str(row["task_name"]),
                prompt=str(row["prompt"]),
                completion=str(row["completion"]),
                total_reward=float(row["total_reward"]),
                passed=bool(row["passed"]),
                per_step_rewards=_parse_per_step(row.get("per_step_rewards")),
                behavior_model=str(row.get("behavior_model") or ""),
                termination_reason=str(row.get("termination_reason") or "unknown"),
                verification_results=_parse_json_list(row.get("verification_results")),
                reward_breakdown=_parse_json_dict(row.get("reward_breakdown")),
            )
        )
    return records


def _parse_per_step(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return []


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    parsed = _parse_json(value)
    return parsed if isinstance(parsed, list) else []


def _parse_json_dict(value: Any) -> dict[str, Any]:
    parsed = _parse_json(value)
    return parsed if isinstance(parsed, dict) else {}


def _parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def load_preferences(path: Path) -> list[PreferenceRecord]:
    """Load chosen/rejected pairs from a `preference_pairs.jsonl` file."""
    records: list[PreferenceRecord] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(
                    PreferenceRecord(
                        task=str(obj.get("task", "")),
                        env_name=str(obj.get("env_name") or _environment_of(_prompt_of(obj["chosen"]))),
                        prompt=_prompt_of(obj["chosen"]),
                        chosen=_assistant_of(obj["chosen"]),
                        rejected=_assistant_of(obj["rejected"]),
                        chosen_reward=float(obj["chosen_reward"]),
                        rejected_reward=float(obj["rejected_reward"]),
                        chosen_passed=bool(obj.get("chosen_passed", False)),
                        rejected_passed=bool(obj.get("rejected_passed", False)),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise MalformedExportError(
                    f"preference_pairs {path} line {lineno} is malformed: {exc}"
                ) from exc
    return records


def _prompt_of(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(messages[0].get("content", "")) if messages else ""


def _assistant_of(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def _environment_of(prompt: str) -> str:
    """Read the environment marker if present."""
    for line in prompt.splitlines():
        if line.startswith("Environment:"):
            return line.partition(":")[2].strip()
    return ""


def load_sft(path_or_dir: Path) -> list[SFTRecord]:
    """Load demonstrations from a Forge export directory, or any file the universal reader accepts."""
    path = Path(path_or_dir)
    if path.is_dir():
        path = _sft_source(path)
        if path is None:
            return []
    if path.suffix == ".parquet":
        return _sft_from_parquet(path)
    if path.suffix == ".json" and _is_forge_json(path):
        return _sft_from_json(path)
    return _sft_from_universal_reader(path)


def _sft_source(directory: Path) -> Path | None:
    """The export in `directory` to read demonstrations from, by priority."""
    candidates = [
        directory / "sft_pairs.jsonl",
        directory / "sft_dataset.jsonl",
        directory / "sft_data.jsonl",
        directory / "batch_export.json",
        directory / "tasks.json",
        directory / "preference_pairs.jsonl",
        directory / "grpo_rollouts.parquet",
    ]
    target_file = next((c for c in candidates if c.exists()), None)
    if target_file is None:
        target_file = next(iter(directory.glob("*.jsonl")), next(iter(directory.glob("*.json")), None))
    return target_file


def _sft_from_parquet(path: Path) -> list[SFTRecord]:
    rollouts = load_rollouts(path)
    valid = [r for r in rollouts if r.passed or r.total_reward > 0]
    return [
        SFTRecord(
            prompt=r.prompt,
            completion=r.completion,
            env_name=r.env_name,
            reward=r.total_reward,
            passed=r.passed,
            behavior_model=r.behavior_model,
        )
        for r in valid
    ]


def _is_forge_json(path: Path) -> bool:
    """Forge batch exports and task files are objects with `sft_items` or `tasks`."""
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and ("sft_items" in data or "tasks" in data)


def _sft_from_json(path: Path) -> list[SFTRecord]:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        raise MalformedExportError(f"could not load json {path}: {exc}") from exc

    records: list[SFTRecord] = []
    if isinstance(data, dict):
        if "sft_items" in data and isinstance(data["sft_items"], list):
            for item in data["sft_items"]:
                records.append(
                    SFTRecord(
                        prompt=str(item.get("prompt", "")),
                        completion=str(item.get("completion", "")),
                        env_name=str(data.get("env_name", "")),
                    )
                )
        elif "tasks" in data and isinstance(data["tasks"], list):
            for task in data["tasks"]:
                prompt = str(task.get("objective") or task.get("prompt") or "")
                golden = task.get("golden", [])
                completion = (
                    "\n".join(
                        f"$ {s.get('command') or s.get('tool', '')}" if isinstance(s, dict) else str(s)
                        for s in golden
                    )
                    if golden
                    else str(task.get("completion", ""))
                )
                records.append(
                    SFTRecord(
                        prompt=prompt,
                        completion=completion,
                        env_name=str(data.get("env_name", "")),
                    )
                )
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                records.append(
                    SFTRecord(
                        prompt=str(item.get("prompt", "")),
                        completion=str(item.get("completion", "")),
                        env_name=str(item.get("env_name", "")),
                    )
                )
    return [r for r in records if r.prompt and r.completion]


def sft_records(records: list[DatasetRecord]) -> list[SFTRecord]:
    """One demonstration per record: first user message as prompt, the first assistant reply after it as completion."""
    demonstrations = []
    for record in records:
        turns = record.messages
        first_user = next((i for i, m in enumerate(turns) if m.role == "user"), None)
        if first_user is None:
            continue
        prompt = turns[first_user].content
        completion = next((m.content for m in turns[first_user + 1:] if m.role == "assistant"), "")
        if not prompt or not completion:
            continue
        meta = record.metadata
        demonstrations.append(
            SFTRecord(
                prompt=prompt,
                completion=completion,
                env_name=str(meta.get("env_name") or _environment_of(prompt)),
                reward=_reward_of(meta),
            )
        )
    return demonstrations


def _reward_of(metadata: dict[str, Any]) -> float:
    value = next((metadata[k] for k in ("reward", "total_reward", "chosen_reward") if k in metadata), 1.0)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MalformedExportError(f"reward {value!r} is not a number") from exc


def _sft_from_universal_reader(path: Path) -> list[SFTRecord]:
    sources = collect_sources(path)
    if sources.documents and not sources.records:
        raise IngestionError(
            f"{path.name} holds raw text only. Benchmark held-out data needs prompt and reference answer pairs."
        )
    if sources.dropped:
        raise MalformedExportError(
            f"{path.name} dropped {sources.dropped} rows and held-out data must load exactly. "
            + " ".join(sources.warnings)
        )
    return sft_records(sources.records)
