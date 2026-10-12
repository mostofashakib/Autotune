"""Mock training backend for testing and simulation without GPUs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from autotune.core.domain.models import CheckpointResult, CleanedDataset, JobStatus, TrainingJob
from autotune.core.ports.trainer import TrainingBackend


class MockTrainingBackend(TrainingBackend):
    """Executes a simulated training run, writing realistic checkpoint artifacts."""

    def __init__(self, step_duration: float = 0.0) -> None:
        self.step_duration = step_duration
        self._cancelled_jobs: set[str] = set()

    def train(self, job: TrainingJob, dataset: CleanedDataset, on_progress=None) -> CheckpointResult:
        out_dir = Path(job.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.step_duration > 0:
            time.sleep(self.step_duration)

        if job.id in self._cancelled_jobs:
            return CheckpointResult(
                job_id=job.id,
                checkpoint_dir=str(out_dir),
                status=JobStatus.CANCELLED,
            )

        # Write dummy model adapter config and manifest
        adapter_config = {
            "base_model_name_or_path": job.recipe.base_model,
            "peft_type": "LORA",
            "quantization": job.recipe.quantization.value,
            "r": job.recipe.lora_r,
            "lora_alpha": job.recipe.lora_alpha,
            "target_modules": job.recipe.target_modules,
        }
        (out_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
        (out_dir / "adapter_model.bin").write_bytes(b"MOCK_ADAPTER_WEIGHTS")

        train_info = {
            "job_id": job.id,
            "samples_trained": len(dataset.train_records),
            "samples_validated": len(dataset.val_records),
            "final_loss": 0.42,
            "epochs": job.recipe.epochs,
        }
        (out_dir / "training_info.json").write_text(json.dumps(train_info, indent=2))

        return CheckpointResult(
            job_id=job.id,
            checkpoint_dir=str(out_dir),
            final_loss=0.42,
            total_steps=100,
            duration_seconds=self.step_duration,
            status=JobStatus.COMPLETED,
        )

    def cancel(self, job_id: str) -> None:
        self._cancelled_jobs.add(job_id)
