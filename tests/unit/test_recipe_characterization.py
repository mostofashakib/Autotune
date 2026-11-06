"""Pin select_recipe output across hardware, dataset size, and task type."""
import json
from pathlib import Path
import pytest
from autotune.core.domain.models import ChatMessage, CleanedDataset, DatasetRecord, TaskType
from autotune.core.services.recipe_selector import AutoRecipeSelector

GOLDEN = Path(__file__).parent / "golden" / "recipe_characterization.json"
VRAMS = (0, 6, 12, 24, 48, 80)
SIZES = (100, 1000, 10000)
TASKS = (TaskType.CHAT, TaskType.TOOL_USE, TaskType.PREFERENCE, TaskType.RL_GRPO)


def _record(i: int, task_type: TaskType) -> DatasetRecord:
    messages = [ChatMessage(role="user", content=f"Hello {i}"), ChatMessage(role="assistant", content=f"World {i}")]
    metadata = {}
    if task_type == TaskType.PREFERENCE:
        metadata = {"chosen": "good", "rejected": "bad"}
    elif task_type == TaskType.RL_GRPO:
        metadata = {"reward": 1.0}
    return DatasetRecord(id=f"r_{i}", messages=messages, metadata=metadata)


def _dataset(count: int, task_type: TaskType) -> CleanedDataset:
    return CleanedDataset(task_type=task_type, total_records=count,
                          train_records=[_record(i, task_type) for i in range(min(count, 5))], val_records=[])


def case_key(vram: float, count: int, task_type: TaskType) -> str:
    return f"{vram}|{count}|{task_type.value}"


def characterize(vram: float, count: int, task_type: TaskType) -> dict:
    try:
        return AutoRecipeSelector().select_recipe(_dataset(count, task_type), available_vram_gb=vram).model_dump(mode="json")
    except ValueError as error:
        return {"error": str(error)}


@pytest.mark.parametrize("task_type", TASKS)
@pytest.mark.parametrize("count", SIZES)
@pytest.mark.parametrize("vram", VRAMS)
def test_select_recipe_matches_golden(vram, count, task_type):
    golden = json.loads(GOLDEN.read_text())
    assert characterize(vram, count, task_type) == golden[case_key(vram, count, task_type)]
