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


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
@pytest.mark.parametrize('kind', [TaskType.CHAT, TaskType.TOOL_USE, TaskType.RL_GRPO])
def test_nested_messages_in_tabular_data(tmp_path, extension, kind):
    import pandas as pd
    messages = [{'role':'user', 'content':'question'}, {'role':'assistant', 'content':'answer'}]
    if kind == TaskType.TOOL_USE:
        messages.insert(1, {'role':'tool', 'content':'result'})
    row = {'messages':json.dumps(messages) if extension == 'csv' else messages}
    if kind == TaskType.RL_GRPO:
        row['reward'] = 1
    frame = pd.DataFrame([row])
    path = tmp_path / ('data.' + extension)
    if extension == 'csv': frame.to_csv(path, index=False)
    else: frame.to_parquet(path, index=False)
    assert AutoFormatSanitizer().sanitize(path).task_type == kind


def _conversation(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    record = AutoFormatSanitizer().sanitize(path, val_split_ratio=0).train_records[0]
    return [(m.role, m.content) for m in record.messages]


@pytest.mark.parametrize("columns", [("user", "assistant"), ("input", "output"), ("query", "response"),
                                     ("question", "response"), ("prompt", "answer")])
def test_prompt_and_response_column_aliases(tmp_path, columns):
    rows = f"{columns[0]},{columns[1]}\nhi,there\n"
    assert _conversation(tmp_path, "rows.csv", rows) == [("user", "hi"), ("assistant", "there")]


def test_alpaca_input_stays_context_when_instruction_present(tmp_path):
    row = json.dumps({"instruction": "Summarize", "input": "Long text", "output": "Short"})
    assert _conversation(tmp_path, "rows.jsonl", row) == [("user", "Summarize\n\nContext: Long text"), ("assistant", "Short")]


def test_sharegpt_conversations(tmp_path):
    row = json.dumps({"conversations": [{"from": "system", "value": "Be brief"}, {"from": "human", "value": "hi"},
                                        {"from": "gpt", "value": "yo"}]})
    assert _conversation(tmp_path, "rows.jsonl", row) == [("system", "Be brief"), ("user", "hi"), ("assistant", "yo")]


@pytest.mark.parametrize("user, assistant", [("human", "bot"), ("user", "model"), ("User", "AI")])
def test_role_aliases(tmp_path, user, assistant):
    row = json.dumps({"messages": [{"role": user, "content": "hi"}, {"role": assistant, "content": "yo"}]})
    assert _conversation(tmp_path, "rows.jsonl", row) == [("user", "hi"), ("assistant", "yo")]


def test_content_parts_join_their_text(tmp_path):
    row = json.dumps({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]},
                                   {"role": "assistant", "content": "yo"}]})
    assert _conversation(tmp_path, "rows.jsonl", row) == [("user", "hi\nthere"), ("assistant", "yo")]


@pytest.mark.parametrize("key", ["data", "rows", "records", "examples", "items", "train"])
def test_json_object_wrapping_rows_is_unwrapped(tmp_path, key):
    rows = json.dumps({"version": 2, key: [{"prompt": "hi", "completion": "yo"}]})
    assert _conversation(tmp_path, "rows.json", rows) == [("user", "hi"), ("assistant", "yo")]


@pytest.mark.parametrize("sep", [";", "|", "\t"])
def test_csv_delimiter_is_detected(tmp_path, sep):
    rows = f"prompt{sep}completion\nhi, friend{sep}yo, pal\n"
    assert _conversation(tmp_path, "rows.csv", rows) == [("user", "hi, friend"), ("assistant", "yo, pal")]
