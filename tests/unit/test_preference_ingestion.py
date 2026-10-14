"""Raw preference data selects DPO without losing either response."""
import json
import pytest
from autotune.core.domain.models import TaskType, TrainingObjective
from autotune.core.services.recipe_selector import AutoRecipeSelector
from autotune.infrastructure.sanitizers.format_normalizer import AutoFormatSanitizer


def sanitize(tmp_path, rows):
    path = tmp_path / 'pairs.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    return AutoFormatSanitizer().sanitize(path)


@pytest.mark.parametrize('messages', [False, True])
def test_preference_strings_and_message_lists_select_dpo(tmp_path, messages):
    chosen, rejected = 'Four', 'Five'
    if messages:
        chosen = [{'role': 'user', 'content': '2+2?'}, {'role': 'assistant', 'content': chosen}]
        rejected = [{'role': 'assistant', 'content': rejected}]
    dataset = sanitize(tmp_path, [{'prompt': '2+2?', 'chosen': chosen, 'rejected': rejected}])
    assert dataset.task_type == TaskType.PREFERENCE
    record = dataset.train_records[0]
    assert record.prompt == '2+2?'
    assert record.metadata['chosen'] == 'Four' and record.metadata['rejected'] == 'Five'
    assert AutoRecipeSelector().select_recipe(dataset, available_vram_gb=24).objective == TrainingObjective.DPO


@pytest.mark.parametrize('chosen,rejected', [(1, 1), (0, 1), ('nan', 1), ('bad', 0)])
def test_unordered_rewards_are_not_preferences(tmp_path, chosen, rejected):
    with pytest.raises(ValueError, match='No valid records'):
        sanitize(tmp_path, [{'prompt':'p', 'chosen':'good', 'rejected':'bad', 'chosen_reward':chosen, 'rejected_reward':rejected}])


@pytest.mark.parametrize('chosen,rejected', [('', 'bad'), (None, 'bad'), ([{'role':'user','content':'not an answer'}], 'bad'), ('same', 'same')])
def test_preference_lookalikes_are_rejected(tmp_path, chosen, rejected):
    with pytest.raises(ValueError, match='No valid records'):
        sanitize(tmp_path, [{'prompt':'p', 'chosen':chosen, 'rejected':rejected}])


def test_precedence_is_independent_of_row_order(tmp_path):
    chat = {'prompt':'hello', 'completion':'hi'}
    tool = {'messages':[{'role':'user','content':'run'}, {'role':'tool','content':'done'}]}
    pref = {'prompt':'p', 'chosen':'good', 'rejected':'bad', 'chosen_reward':2, 'rejected_reward':1}
    reward = {'prompt':'q', 'completion':'a', 'reward':1}
    for rows, kind in [([chat, tool, pref], TaskType.PREFERENCE), ([pref, tool, chat], TaskType.PREFERENCE), ([reward, pref, tool], TaskType.RL_GRPO)]:
        assert sanitize(tmp_path, rows).task_type == kind


def test_dpo_dataset_adapter_rejects_mixed_non_preferences(tmp_path):
    from autotune.infrastructure.backends.trl_peft_backend import training_rows
    dataset = sanitize(tmp_path, [{'prompt':'p','chosen':'good','rejected':'bad'}, {'prompt':'q','completion':'a'}])
    with pytest.raises(ValueError, match='preference'):
        training_rows(dataset.train_records + dataset.val_records, TrainingObjective.DPO)


def test_dpo_dataset_adapter_preserves_rejected_response(tmp_path):
    from autotune.infrastructure.backends.trl_peft_backend import training_rows
    dataset = sanitize(tmp_path, [{'prompt':'p','chosen':'good','rejected':'bad'}])
    assert training_rows(dataset.train_records, TrainingObjective.DPO) == [{'prompt':'p','chosen':'good','rejected':'bad'}]
