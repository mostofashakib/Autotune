"""Catalog fit decisions depend on size as well as hardware."""
import pytest
from fastapi.testclient import TestClient
from autotune.core.domain.models import CleanedDataset, TaskType, QuantizationMethod
from autotune.core.services.recipe_selector import AutoRecipeSelector
from autotune.presentation.api.app import create_app


def dataset(task=TaskType.CHAT):
    return CleanedDataset(task_type=task, total_records=100, train_records=[], val_records=[])


def test_fourteen_billion_model_uses_qlora_on_24gb():
    recipe = AutoRecipeSelector().select_recipe(dataset(), forced_model='Qwen/Qwen2.5-14B-Instruct', available_vram_gb=24)
    assert recipe.quantization == QuantizationMethod.QLORA_4BIT_NF4


def test_small_model_uses_lora_on_same_gpu():
    recipe = AutoRecipeSelector().select_recipe(dataset(), forced_model='Qwen/Qwen2.5-1.5B-Instruct', available_vram_gb=24)
    assert recipe.quantization == QuantizationMethod.LORA_16BIT


def test_model_that_does_not_fit_is_rejected():
    with pytest.raises(ValueError, match='QLoRA'):
        AutoRecipeSelector().select_recipe(dataset(), forced_model='Qwen/Qwen2.5-14B-Instruct', available_vram_gb=8)


def test_forced_precision_cannot_bypass_fit_check():
    with pytest.raises(ValueError, match='VRAM'):
        AutoRecipeSelector().select_recipe(dataset(), forced_model='Qwen/Qwen2.5-14B-Instruct', available_vram_gb=24, forced_quantization=QuantizationMethod.LORA_16BIT)


def test_catalog_endpoint_lists_all_and_marks_fit():
    response = TestClient(create_app()).get('/api/models?vram=24')
    assert response.status_code == 200
    rows = response.json()
    assert 8 <= len(rows) <= 12
    large = next(row for row in rows if row['hf_id'] == 'Qwen/Qwen2.5-14B-Instruct')
    assert large['fits'] and large['quantization'] == 'qlora_4bit_nf4'
    assert any(row['gated'] for row in rows)
    low = TestClient(create_app()).get('/api/models?vram=8').json()
    assert any(not row['fits'] for row in low)


@pytest.mark.parametrize('vram', ['-1', 'nan', 'inf'])
def test_invalid_vram_is_not_a_fit(vram):
    assert TestClient(create_app()).get('/api/models?vram=' + vram).status_code == 422


def test_unknown_size_is_not_guessed_from_name():
    with pytest.raises(ValueError, match='parameter'):
        AutoRecipeSelector().select_recipe(dataset(), forced_model='someone/fake-1B', available_vram_gb=24)


def test_tiny_gpu_selects_small_model():
    from autotune.core.domain.model_catalog import get_model
    recipe = AutoRecipeSelector().select_recipe(dataset(TaskType.TOOL_USE), available_vram_gb=8)
    model = get_model(recipe.base_model)
    assert model.supports_tool_calling
    assert model.qlora_min_vram_gb <= 8
