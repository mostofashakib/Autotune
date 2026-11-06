import pytest
from autotune.core.domain.model_catalog import MODELS, ModelSpec, get_model
from autotune.core.domain.models import (
    ChatMessage, CleanedDataset, DatasetRecord, QuantizationMethod, TaskType, TrainingObjective,
)
from autotune.core.services.training_planner import TrainingPlanner, TrainingRequest


class FakeProbe:
    def __init__(self, vram: float) -> None:
        self.vram = vram

    def available_vram_gb(self) -> float:
        return self.vram


def fake_resolve(hf_id: str) -> ModelSpec:
    model = get_model(hf_id)
    assert model is not None, hf_id
    return model


def chat_dataset(count: int = 100) -> CleanedDataset:
    records = [DatasetRecord(id=f"r{i}", messages=[ChatMessage(role="user", content="hi"),
                                                   ChatMessage(role="assistant", content="yo")])
               for i in range(count)]
    return CleanedDataset(task_type=TaskType.CHAT, total_records=count, train_records=records, val_records=[])


def planner(vram: float = 48.0) -> TrainingPlanner:
    return TrainingPlanner(FakeProbe(vram), fake_resolve)


def test_recommend_and_plan_honor_the_same_forced_quantization():
    request = TrainingRequest(base_model=MODELS[0].hf_id, quantization=QuantizationMethod.LORA_16BIT)
    assert planner().recommend(chat_dataset(), request).recipe.quantization == QuantizationMethod.LORA_16BIT
    assert planner().plan(chat_dataset(), request).recipe.quantization == QuantizationMethod.LORA_16BIT


def test_plan_applies_intensity_then_overrides():
    quick = planner().plan(chat_dataset(), TrainingRequest(intensity="quick"))
    plan = planner().plan(chat_dataset(), TrainingRequest(intensity="quick", overrides={"epochs": 7}))
    assert plan.recipe.epochs == 7
    assert plan.recipe.learning_rate == quick.recipe.learning_rate
    assert plan.model.hf_id == plan.recipe.base_model


def test_plan_rejects_unknown_override():
    with pytest.raises(ValueError, match="warmup_ratio"):
        planner().plan(chat_dataset(), TrainingRequest(overrides={"warmup_ratio": 0.1}))


def test_plan_rejects_dpo_without_pairs():
    with pytest.raises(ValueError, match="DPO requires a prompt and distinct chosen/rejected"):
        planner().plan(chat_dataset(), TrainingRequest(objective=TrainingObjective.DPO))


def test_presets_cover_three_intensities():
    plan = planner().recommend(chat_dataset(), TrainingRequest())
    assert set(plan.presets) == {"quick", "balanced", "thorough"}
    assert plan.vram_gb == 48.0


def test_plan_model_matches_recipe_after_smaller_model_fallback():
    largest_fitting = "Qwen/Qwen2.5-7B-Instruct"
    plan = planner(12.0).plan(chat_dataset(), TrainingRequest())
    assert plan.recipe.base_model != largest_fitting
    assert plan.model.hf_id == plan.recipe.base_model


def test_plan_validates_the_overridden_recipe():
    request = TrainingRequest(overrides={"batch_size": 4096})
    assert planner().recommend(chat_dataset(), request).recipe.batch_size != 4096
    with pytest.raises(ValueError, match="exceeds"):
        planner().plan(chat_dataset(), request)
