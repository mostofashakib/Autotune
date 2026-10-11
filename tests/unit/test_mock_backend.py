import json
from pathlib import Path
from autotune.core.domain.models import (
    ChatMessage,
    CleanedDataset,
    DatasetRecord,
    JobStatus,
    Recipe,
    TaskType,
    TrainingJob,
)
from autotune.infrastructure.backends.mock_backend import MockTrainingBackend


def test_mock_backend_produces_checkpoint(tmp_path: Path):
    out_dir = tmp_path / "model_weights"
    backend = MockTrainingBackend()

    records = [
        DatasetRecord(
            id="1",
            messages=[
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="Hello"),
            ],
        )
    ]
    dataset = CleanedDataset(
        task_type=TaskType.CHAT,
        total_records=1,
        train_records=records,
        val_records=[],
    )

    recipe = Recipe(base_model="meta-llama/Llama-3.1-8B-Instruct")
    job = TrainingJob(
        id="job_001",
        dataset_path="data.jsonl",
        output_dir=str(out_dir),
        recipe=recipe,
    )

    result = backend.train(job, dataset)

    assert result.status == JobStatus.COMPLETED
    assert result.job_id == "job_001"
    assert (out_dir / "adapter_config.json").exists()
    assert (out_dir / "adapter_model.bin").exists()

    config = json.loads((out_dir / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert config["quantization"] == "qlora_4bit_nf4"


def test_mock_backend_cancellation(tmp_path: Path):
    out_dir = tmp_path / "model_cancel"
    backend = MockTrainingBackend()

    records = [DatasetRecord(id="1", messages=[ChatMessage(role="user", content="Hi"), ChatMessage(role="assistant", content="Hello")])]
    dataset = CleanedDataset(task_type=TaskType.CHAT, total_records=1, train_records=records, val_records=[])
    job = TrainingJob(id="job_to_cancel", dataset_path="data.jsonl", output_dir=str(out_dir), recipe=Recipe(base_model="test"))

    backend.cancel("job_to_cancel")
    result = backend.train(job, dataset)
    assert result.status == JobStatus.CANCELLED
