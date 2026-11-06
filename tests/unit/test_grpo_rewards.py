"""GRPO trains on a reference reward built from the highest-reward completion per prompt."""
import pytest

from autotune.core.domain.models import ChatMessage, DatasetRecord
from autotune.infrastructure.backends.grpo_rewards import grpo_rows, reference_reward


def record(prompt, completion, **metadata):
    return DatasetRecord(messages=[ChatMessage(role="user", content=prompt),
                                   ChatMessage(role="assistant", content=completion)], metadata=metadata)


def test_reference_reward_scores_token_f1():
    assert reference_reward(["the cat sat"], ["the cat sat"]) == [1.0]
    assert reference_reward(["dog"], ["the cat sat"]) == [0.0]


def test_reference_reward_reads_conversational_completions():
    assert reference_reward([[{"role": "assistant", "content": "the cat sat"}]], ["the cat sat"]) == [1.0]


def test_grpo_rows_keep_the_best_completion_per_prompt():
    rows = grpo_rows([record("p", "weak", total_reward=0.2), record("p", "strong", total_reward=0.9)])
    assert rows == [{"prompt": [{"role": "user", "content": "p"}], "reference": "strong"}]


def test_grpo_reference_breaks_ties_by_first_row():
    rows = grpo_rows([record("p", "first", reward=1.0), record("p", "second", reward=1.0)])
    assert rows[0]["reference"] == "first"


def test_grpo_rewards_list_is_summed():
    rows = grpo_rows([record("p", "low", rewards=[0.1, 0.2]), record("p", "high", rewards=[0.5, 0.5])])
    assert rows[0]["reference"] == "high"


def test_grpo_rows_reject_records_without_rewards():
    with pytest.raises(ValueError, match="GRPO needs a numeric reward"):
        grpo_rows([record("p", "answer")])
