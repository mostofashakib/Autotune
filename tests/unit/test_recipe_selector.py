import pytest
from autotune.core.domain.models import CleanedDataset, DatasetRecord, ChatMessage, QuantizationMethod, TaskType
from autotune.core.services.recipe_selector import AutoRecipeSelector


def make_dataset(count: int, task_type: TaskType = TaskType.CHAT) -> CleanedDataset:
    records = [
        DatasetRecord(
            id=f"r_{i}",
            messages=[
                ChatMessage(role="user", content=f"Hello {i}"),
                ChatMessage(role="assistant", content=f"World {i}"),
            ],
        )
        for i in range(count)
    ]
    return CleanedDataset(
        task_type=task_type,
        total_records=count,
        train_records=records,
        val_records=[],
    )


def test_qlora_selected_for_consumer_vram():
    selector = AutoRecipeSelector()
    ds = make_dataset(100)
    # 24GB RTX 3090 / 4090
    recipe = selector.select_recipe(ds, available_vram_gb=24.0)
    assert recipe.quantization == QuantizationMethod.QLORA_4BIT_NF4


def test_lora_selected_for_datacenter_vram():
    selector = AutoRecipeSelector()
    ds = make_dataset(100)
    # 80GB A100 / H100
    recipe = selector.select_recipe(ds, available_vram_gb=80.0)
    assert recipe.quantization == QuantizationMethod.LORA_16BIT


def test_small_dataset_gets_higher_rank():
    selector = AutoRecipeSelector()
    ds = make_dataset(200)
    recipe = selector.select_recipe(ds, available_vram_gb=24.0)
    assert recipe.lora_r == 32
    assert recipe.lora_alpha == 64
    assert recipe.epochs == 4


def test_large_dataset_gets_standard_rank_and_fewer_epochs():
    selector = AutoRecipeSelector()
    ds = make_dataset(10000)
    recipe = selector.select_recipe(ds, available_vram_gb=24.0)
    assert recipe.lora_r == 16
    assert recipe.epochs == 2
    assert recipe.learning_rate == 3e-4


def test_tool_use_selects_qwen_model():
    selector = AutoRecipeSelector()
    ds = make_dataset(500, task_type=TaskType.TOOL_USE)
    recipe = selector.select_recipe(ds, available_vram_gb=24.0)
    assert "Qwen" in recipe.base_model
