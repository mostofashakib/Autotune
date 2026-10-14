"""Preset scaling and preflight checks keep advanced overrides accountable."""
import pytest
from fastapi.testclient import TestClient
from autotune.core.domain.models import Recipe, QuantizationMethod, TrainingObjective
from autotune.presentation.api import app as api


def recipe(**kwargs):
    return Recipe(base_model='Qwen/Qwen2.5-14B-Instruct', **kwargs)


def test_presets_are_scaled_from_recommendation_without_mutation():
    from autotune.core.domain.training_settings import intensity_recipes
    base = recipe(epochs=4, learning_rate=1e-4)
    presets = intensity_recipes(base)
    assert presets['quick'].epochs == 2 and presets['quick'].learning_rate == 5e-5
    assert presets['balanced'] == base
    assert presets['thorough'].epochs == 6 and presets['thorough'].learning_rate == 7.5e-5
    assert base.epochs == 4


def test_presets_never_exceed_grpo_learning_rate_cap():
    from autotune.core.domain.training_settings import intensity_recipes
    assert all(r.learning_rate <= 2e-5 for r in intensity_recipes(recipe(objective=TrainingObjective.GRPO, learning_rate=1e-4)).values())


def test_memory_grows_with_batch_sequence_and_model_size():
    from autotune.core.domain.training_settings import estimate_memory_gb
    from autotune.core.domain.model_catalog import get_model
    model = get_model('Qwen/Qwen2.5-14B-Instruct')
    base = estimate_memory_gb(recipe(batch_size=1), model)
    assert estimate_memory_gb(recipe(batch_size=8), model) > base
    assert estimate_memory_gb(recipe(batch_size=1, max_seq_length=8192), model) > base
    assert estimate_memory_gb(recipe(batch_size=1, quantization=QuantizationMethod.LORA_16BIT), model) > base


def test_memory_validation_names_changes_to_make():
    from autotune.core.domain.training_settings import validate_training_recipe
    from autotune.core.domain.model_catalog import get_model
    with pytest.raises(ValueError, match='batch size.*sequence length'):
        validate_training_recipe(recipe(batch_size=32), get_model('Qwen/Qwen2.5-14B-Instruct'), 24)


def test_low_memory_recipe_is_accepted():
    from autotune.core.domain.training_settings import validate_training_recipe
    from autotune.core.domain.model_catalog import get_model
    validate_training_recipe(recipe(batch_size=1), get_model('Qwen/Qwen2.5-14B-Instruct'), 24)


@pytest.mark.parametrize('overrides', [{'batch_size':0}, {'max_seq_length':-1}, {'learning_rate':0}, {'epochs':0}, {'lora_r':0}, {'gradient_accumulation_steps':0}, {'objective':'invalid'}, {'quantization':'invalid'}, {'intensity':'extreme'}])
def test_invalid_overrides_are_http_errors_before_launch(tmp_path, monkeypatch, overrides):
    path = tmp_path / 'data.jsonl'
    path.write_text('{"prompt":"p", "completion":"a"}')
    monkeypatch.setattr(api, 'build_training_backend', lambda: pytest.fail('must reject before launch'))
    response = TestClient(api.create_app()).post('/api/training/start', json={'dataset_name':str(path), **overrides})
    assert response.status_code == 422, response.text


def test_memory_override_is_rejected_by_api(tmp_path, monkeypatch):
    path = tmp_path / 'data.jsonl'
    path.write_text('{"prompt":"p", "completion":"a"}')
    monkeypatch.setattr(api.AutoRecipeSelector, 'detect_available_vram_gb', staticmethod(lambda: 24))
    monkeypatch.setattr(api, 'build_training_backend', lambda: pytest.fail('must reject before launch'))
    response = TestClient(api.create_app()).post('/api/training/start', json={'dataset_name':str(path), 'batch_size':64})
    assert response.status_code == 422 and 'sequence length' in response.text


def test_recommendation_returns_domain_presets(tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text('{"prompt":"p", "completion":"a"}')
    response = TestClient(api.create_app()).post('/api/recipe/recommend', json={'dataset_name':str(path)})
    assert response.status_code == 200
    body = response.json()
    assert body['presets']['quick']['epochs'] < body['recipe']['epochs']
    assert body['presets']['balanced'] == body['recipe']


def test_selector_uses_qlora_when_lora_activations_do_not_fit():
    from autotune.core.services.recipe_selector import AutoRecipeSelector
    from autotune.core.domain.models import CleanedDataset, TaskType
    ds = CleanedDataset(task_type=TaskType.CHAT, total_records=100, train_records=[], val_records=[])
    result = AutoRecipeSelector().select_recipe(ds, forced_model='Qwen/Qwen2.5-1.5B-Instruct', available_vram_gb=8)
    assert result.quantization == QuantizationMethod.QLORA_4BIT_NF4


def test_uppercase_objective_stays_compatible():
    assert api.TrainingStartRequest(dataset_name='x', objective='SFT').objective == TrainingObjective.SFT


def test_dpo_rejects_non_preference_rows_before_start(tmp_path, monkeypatch):
    path = tmp_path / 'data.jsonl'
    path.write_text('{"prompt":"p", "completion":"a"}')
    monkeypatch.setattr(api, 'build_training_backend', lambda: pytest.fail('must reject before launch'))
    response = TestClient(api.create_app()).post('/api/training/start', json={'dataset_name':str(path), 'objective':'dpo'})
    assert response.status_code == 422 and 'preference' in response.text


def test_final_overrides_are_applied_before_memory_validation(tmp_path, monkeypatch):
    from unittest.mock import Mock
    path = tmp_path / 'data.jsonl'
    path.write_text('{"prompt":"p", "completion":"a"}')
    monkeypatch.setattr(api.AutoRecipeSelector, 'detect_available_vram_gb', staticmethod(lambda:20))
    backend = Mock()
    backend.train.side_effect = RuntimeError('stop')
    monkeypatch.setattr(api, 'build_training_backend', lambda:backend)
    response = TestClient(api.create_app()).post('/api/training/start', json={
        'dataset_name':str(path), 'base_model':'Qwen/Qwen2.5-14B-Instruct', 'lora_r':16, 'batch_size':1})
    assert response.status_code == 202, response.text


def test_vram_detection_uses_free_cuda_memory(monkeypatch):
    import sys
    from types import SimpleNamespace
    from autotune.core.services.recipe_selector import AutoRecipeSelector
    cuda = SimpleNamespace(is_available=lambda:True, current_device=lambda:0,
                           get_device_properties=lambda _:SimpleNamespace(total_memory=24 * 1024**3),
                           mem_get_info=lambda *_:(10 * 1024**3,24 * 1024**3))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    assert AutoRecipeSelector.detect_available_vram_gb() == 10


def test_mps_is_not_reported_as_a_cuda_training_gpu(monkeypatch):
    import sys
    from types import SimpleNamespace
    from autotune.core.services.recipe_selector import AutoRecipeSelector
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False), backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:True)))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    assert AutoRecipeSelector.detect_available_vram_gb() == 0
