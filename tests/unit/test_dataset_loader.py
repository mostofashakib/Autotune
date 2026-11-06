"""Dataset loader tests for rollouts and preference pairs."""
from __future__ import annotations

import json
import pandas as pd
import pytest
from autotune.infrastructure.sanitizers.dataset_loader import (
    MalformedExportError,
    load_preferences,
    load_rollouts,
    load_sft,
)
from autotune.core.ports.sanitizer import IngestionError


def _write_rollouts(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_load_rollouts_round_trips_grpo_parquet(tmp_path):
    path = tmp_path / "grpo_rollouts.parquet"
    _write_rollouts(
        path,
        [
            {
                "episode_id": "ep1",
                "env_name": "e",
                "task_name": "t",
                "seed": 0,
                "agent_id": "random",
                "prompt": "Task: t",
                "completion": "$ ls",
                "total_reward": 0.8,
                "passed": True,
                "total_steps": 3,
                "per_step_rewards": json.dumps([0.1, 0.3, 0.4]),
                "behavior_model": "vllm:model",
                "termination_reason": "submitted",
                "verification_results": json.dumps([{"passed": True}]),
                "reward_breakdown": json.dumps({"total_reward": 0.8}),
            }
        ],
    )
    records = load_rollouts(path)
    assert len(records) == 1
    r = records[0]
    assert r.episode_id == "ep1" and r.task_name == "t"
    assert r.total_reward == 0.8 and r.passed is True
    assert r.per_step_rewards == [0.1, 0.3, 0.4]
    assert r.behavior_model == "vllm:model"
    assert r.termination_reason == "submitted"
    assert r.verification_results == [{"passed": True}]
    assert r.reward_breakdown == {"total_reward": 0.8}


def test_load_rollouts_rejects_missing_columns(tmp_path):
    path = tmp_path / "grpo_rollouts.parquet"
    _write_rollouts(path, [{"episode_id": "ep1", "total_reward": 0.5}])
    with pytest.raises(MalformedExportError):
        load_rollouts(path)


def test_load_preferences_parses_chosen_rejected(tmp_path):
    path = tmp_path / "preference_pairs.jsonl"
    record = {
        "chosen": [
            {"role": "user", "content": "Task: t"},
            {"role": "assistant", "content": "$ good"},
        ],
        "rejected": [
            {"role": "user", "content": "Task: t"},
            {"role": "assistant", "content": "$ bad"},
        ],
        "chosen_reward": 1.0,
        "rejected_reward": 0.0,
        "task": "t",
        "chosen_passed": True,
        "rejected_passed": False,
    }
    path.write_text(json.dumps(record) + "\n")
    prefs = load_preferences(path)
    assert len(prefs) == 1
    p = prefs[0]
    assert p.prompt == "Task: t"
    assert p.chosen == "$ good" and p.rejected == "$ bad"
    assert p.chosen_reward == 1.0 and p.rejected_reward == 0.0


def test_load_preferences_rejects_malformed_line(tmp_path):
    path = tmp_path / "preference_pairs.jsonl"
    path.write_text('{"chosen": [], "rejected": []}\n')
    with pytest.raises(MalformedExportError):
        load_preferences(path)


def test_load_sft_reads_csv_question_answer(tmp_path):
    path = tmp_path / "heldout.csv"
    path.write_text("question,answer\nWhat is 2+2?,4\n")
    records = load_sft(path)
    assert [(r.prompt, r.completion) for r in records] == [("What is 2+2?", "4")]


def test_load_sft_reads_sharegpt_jsonl(tmp_path):
    path = tmp_path / "heldout.jsonl"
    row = {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}
    path.write_text(json.dumps(row) + "\n")
    records = load_sft(path)
    assert [(r.prompt, r.completion) for r in records] == [("Hi", "Hello")]


def test_load_sft_keeps_reward_and_environment(tmp_path):
    path = tmp_path / "sft.jsonl"
    row = {"prompt": "Environment: shell\nList files", "completion": "ls", "reward": 0.5}
    path.write_text(json.dumps(row) + "\n")
    (record,) = load_sft(path)
    assert record.env_name == "shell" and record.reward == 0.5


def test_load_sft_rejects_raw_text(tmp_path):
    path = tmp_path / "notes.csv"
    path.write_text("text\nSome free text with no answer\n")
    with pytest.raises(IngestionError, match="prompt and reference answer pairs"):
        load_sft(path)


FORGE_PAIR = {
    "chosen": [{"role": "user", "content": "Task: t"}, {"role": "assistant", "content": "$ good"}],
    "rejected": [{"role": "user", "content": "Task: t"}, {"role": "assistant", "content": "$ bad"}],
    "chosen_reward": 0.9,
    "rejected_reward": 0.1,
}


def test_load_sft_reads_forge_preference_directory(tmp_path):
    (tmp_path / "preference_pairs.jsonl").write_text(json.dumps(FORGE_PAIR) + "\n")
    (record,) = load_sft(tmp_path)
    assert (record.prompt, record.completion, record.reward) == ("Task: t", "$ good", 0.9)


def test_load_sft_rejects_malformed_lines(tmp_path):
    path = tmp_path / "heldout.jsonl"
    path.write_text('{"prompt": "q", "completion": "a"}\n{broken\n')
    with pytest.raises(MalformedExportError, match="line"):
        load_sft(path)


def test_load_sft_uses_first_reply_of_multi_turn_row(tmp_path):
    path = tmp_path / "heldout.jsonl"
    messages = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    path.write_text(json.dumps({"messages": messages}) + "\n")
    (record,) = load_sft(path)
    assert (record.prompt, record.completion) == ("q1", "a1")
