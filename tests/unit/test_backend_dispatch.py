"""Trainer dispatch uses conversational data and the configured training settings."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from autotune.core.domain.models import CleanedDataset, DatasetRecord, ChatMessage, Recipe, QuantizationMethod, TrainingObjective, TrainingJob, TaskType
from autotune.infrastructure.backends.trl_peft_backend import TrlPeftBackend


@pytest.fixture
def runtime(monkeypatch):
    seen = {}
    model = Mock()
    tokenizer = Mock(pad_token='pad')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True, is_bf16_supported=lambda:False)))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw:model),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw:tokenizer), TrainerCallback=object))
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(LoraConfig=lambda **kw:kw, get_peft_model=lambda *a:model, prepare_model_for_kbit_training=lambda m:m))
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows:rows)))
    def config(*, max_length, **kw):
        assert 'max_seq_length' not in kw and 'evaluation_strategy' not in kw
        return {'max_length':max_length, **kw}
    def trainer(kind, *, processing_class, **kw):
        assert 'tokenizer' not in kw
        seen.update(kind=kind, **kw)
        return SimpleNamespace(train=lambda:SimpleNamespace(training_loss=.2, global_step=1))
    trl = SimpleNamespace(SFTConfig=config, DPOConfig=config,
                          SFTTrainer=lambda **kw:trainer('sft', **kw), DPOTrainer=lambda **kw:trainer('dpo', **kw))
    monkeypatch.setitem(sys.modules, 'trl', trl)
    return seen, trl


def run(tmp_path, objective):
    recipe = Recipe(base_model='Qwen/Qwen2.5-1.5B-Instruct', quantization=QuantizationMethod.NONE, objective=objective,
                    epochs=5, batch_size=3, gradient_accumulation_steps=7, max_seq_length=1024)
    record = DatasetRecord(prompt='p', messages=[ChatMessage(role='user',content='p'),ChatMessage(role='assistant',content='good')], metadata={'chosen':'good','rejected':'bad'})
    dataset = CleanedDataset(task_type=TaskType.PREFERENCE, total_records=1, train_records=[record], val_records=[])
    return TrlPeftBackend().train(TrainingJob(id='test', dataset_path='data',output_dir=str(tmp_path/'out'),recipe=recipe), dataset)


@pytest.mark.parametrize('objective', [TrainingObjective.SFT, TrainingObjective.DPO])
def test_dispatch_honors_advanced_settings(tmp_path, runtime, objective):
    seen, _ = runtime
    result = run(tmp_path, objective)
    assert result.total_steps == 1 and seen['kind'] == objective.value
    assert seen['args']['max_length'] == 1024
    assert seen['args']['per_device_train_batch_size'] == 3
    assert seen['args']['gradient_accumulation_steps'] == 7
    assert seen['args']['num_train_epochs'] == 5
    if objective == TrainingObjective.DPO:
        assert seen['train_dataset'][0]['rejected'] == [{'role':'assistant','content':'bad'}]


def test_grpo_error_does_not_fall_back_to_sft(tmp_path, runtime):
    _, trl = runtime
    trl.GRPOConfig = lambda **kw:kw
    def broken(**kw): raise RuntimeError('reward configuration missing')
    trl.GRPOTrainer = broken
    with pytest.raises(RuntimeError, match='reward configuration missing'):
        run(tmp_path, TrainingObjective.GRPO)


def test_no_gpu_fails_before_model_download(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(sys.modules['torch'].cuda, 'is_available', lambda:False)
    sys.modules['transformers'].AutoModelForCausalLM.from_pretrained = lambda *a, **kw:pytest.fail('must not download')
    with pytest.raises(RuntimeError, match='CUDA GPU'):
        run(tmp_path, TrainingObjective.SFT)
