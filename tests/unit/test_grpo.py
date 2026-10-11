import pytest
from autotune.core.domain.models import CleanedDataset, DatasetRecord, ChatMessage, TaskType, TrainingObjective
from autotune.core.services.recipe_selector import AutoRecipeSelector


def make_rl_dataset(count: int) -> CleanedDataset:
    records = [
        DatasetRecord(
            id=f"rl_{i}",
            prompt=f"Solve math problem {i}",
            messages=[
                ChatMessage(role="user", content=f"Solve math problem {i}"),
                ChatMessage(role="assistant", content=f"Step 1: Compute... {i}"),
            ],
            metadata={"reward_target": 1.0},
        )
        for i in range(count)
    ]
    return CleanedDataset(
        task_type=TaskType.RL_GRPO,
        total_records=count,
        train_records=records,
        val_records=[],
    )


def test_rl_grpo_task_selects_grpo_objective():
    selector = AutoRecipeSelector()
    dataset = make_rl_dataset(100)
    recipe = selector.select_recipe(dataset, available_vram_gb=48.0)
    assert recipe.objective == TrainingObjective.GRPO
    assert recipe.grpo_group_size == 4
    assert recipe.grpo_epsilon == 0.2
    assert recipe.learning_rate <= 2e-5


def test_forced_grpo_objective():
    selector = AutoRecipeSelector()
    dataset = make_rl_dataset(50)
    recipe = selector.select_recipe(dataset, forced_objective=TrainingObjective.GRPO)
    assert recipe.objective == TrainingObjective.GRPO
    assert "Qwen" in recipe.base_model
