import json
from pathlib import Path
import pytest
from autotune.infrastructure.sanitizers.format_normalizer import AutoFormatSanitizer
from autotune.core.domain.models import TaskType


def test_sanitize_messages_jsonl(tmp_path: Path):
    file_path = tmp_path / "data.jsonl"
    lines = [
        {"messages": [{"role": "user", "content": "How do I reverse a string?"}, {"role": "assistant", "content": "Use s[::-1]"}]},
        {"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "A high-level programming language."}]},
        {"corrupt": "data"},  # should be dropped
    ]
    file_path.write_text("\n".join(json.dumps(l) for l in lines))

    sanitizer = AutoFormatSanitizer()
    dataset = sanitizer.sanitize(file_path, val_split_ratio=0.5)

    assert dataset.total_records == 2
    assert dataset.dropped_count == 1
    assert len(dataset.train_records) == 1
    assert len(dataset.val_records) == 1
    assert dataset.train_records[0].messages[0].content == "How do I reverse a string?"


def test_sanitize_prompt_completion_pairs(tmp_path: Path):
    file_path = tmp_path / "pairs.jsonl"
    lines = [
        {"prompt": "Hello", "completion": "Hi there!"},
        {"prompt": "Calculate 2+2", "completion": "4"},
    ]
    file_path.write_text("\n".join(json.dumps(l) for l in lines))

    sanitizer = AutoFormatSanitizer()
    dataset = sanitizer.sanitize(file_path)

    assert dataset.total_records == 2
    assert dataset.train_records[0].messages[0].role == "user"
    assert dataset.train_records[0].messages[1].role == "assistant"


def test_sanitize_csv(tmp_path: Path):
    file_path = tmp_path / "dataset.csv"
    file_path.write_text("prompt,completion\nWrite hello,print('hello')\nWrite bye,print('bye')\n")

    sanitizer = AutoFormatSanitizer()
    dataset = sanitizer.sanitize(file_path)

    assert dataset.total_records == 2
    assert dataset.train_records[0].messages[1].content == "print('hello')"


def test_detects_tool_use_task_type(tmp_path: Path):
    file_path = tmp_path / "tool_data.jsonl"
    lines = [
        {
            "messages": [
                {"role": "user", "content": "Check weather in Tokyo"},
                {"role": "tool", "content": '{"temp": "18C"}'},
                {"role": "assistant", "content": "It is currently 18C in Tokyo."},
            ]
        }
    ]
    file_path.write_text("\n".join(json.dumps(l) for l in lines))

    sanitizer = AutoFormatSanitizer()
    dataset = sanitizer.sanitize(file_path)

    assert dataset.task_type == TaskType.TOOL_USE
