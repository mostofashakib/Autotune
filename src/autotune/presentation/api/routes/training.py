"""Recommend a recipe, then start, watch, and cancel training runs."""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from autotune.core.domain.models import (
    CleanedDataset,
    JobStatus,
    QuantizationMethod,
    TrainingJob,
    TrainingObjective,
    TrainingProgress,
)
from autotune.core.domain.training_settings import TrainingIntensity
from autotune.core.ports.trainer import TrainingBackend
from autotune.core.services.training_planner import TUNABLE_FIELDS, TrainingRequest
from autotune.defaults import CHECKPOINT_DIR
from autotune.presentation.api.dataset_ingestion import load_dataset
from autotune.presentation.api.jobs import JobHandle, summary
from autotune.presentation.api.routes.datasets import presets_payload, resolve_dataset
from autotune.presentation.api.schemas import JobStarted, RecipePresets, TrainingCancelling, TrainingRun, TrainingSummary
from autotune.presentation.api.state import State
from autotune.settings import HF_TOKEN, require_credential

router = APIRouter()


class TrainingStartRequest(BaseModel):
    dataset_name: str
    base_model: Optional[str] = None
    objective: TrainingObjective | None = None
    quantization: QuantizationMethod | None = None
    intensity: TrainingIntensity = "balanced"
    lora_r: int | None = Field(default=None, ge=1, le=512)
    lora_alpha: int | None = Field(default=None, ge=1, le=1024)
    learning_rate: float | None = Field(default=None, gt=0, le=.1, allow_inf_nan=False)
    epochs: int | None = Field(default=None, ge=1, le=100)
    batch_size: int | None = Field(default=None, ge=1, le=1024)
    gradient_accumulation_steps: int | None = Field(default=None, ge=1, le=4096)
    max_seq_length: int | None = Field(default=None, ge=128, le=131072)
    output_dir: str = CHECKPOINT_DIR

    @field_validator("objective", mode="before")
    @classmethod
    def normalize_objective(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value


def _training_request(req: TrainingStartRequest) -> TrainingRequest:
    fields = req.model_dump()
    return TrainingRequest(
        base_model=req.base_model, objective=req.objective, quantization=req.quantization,
        intensity=req.intensity,
        overrides={name: fields[name] for name in TUNABLE_FIELDS if fields[name] is not None})


def _progress_recorder(handle: JobHandle) -> Callable[[TrainingProgress], None]:
    """Records each progress update the trainer logs on the job."""
    history: list[float] = []

    def record(progress: TrainingProgress) -> None:
        fields: dict[str, Any] = {"current_step": progress.step, "total_steps": progress.total_steps}
        if progress.total_steps:
            fields["progress_percent"] = min(100, int(progress.step * 100 / progress.total_steps))
        if progress.loss is not None:
            history.append(progress.loss)
            fields.update(current_loss=progress.loss, loss_history=list(history))
        handle.update(**fields)
        if progress.loss is not None:
            handle.log(f"Step {progress.step}/{progress.total_steps}: loss {progress.loss:.4f}")

    return record


def _train(handle: JobHandle, backend: TrainingBackend, job: TrainingJob, dataset: CleanedDataset) -> dict[str, Any]:
    try:
        result = backend.train(job, dataset, on_progress=_progress_recorder(handle))
    except Exception as exc:
        handle.log(f"Training failed: {exc}")
        raise
    if result.status == JobStatus.CANCELLED:
        handle.log("Training stopped after the cancel request.")
        return {"status": "cancelled"}
    handle.log(f"Training completed. Checkpoint saved to {result.checkpoint_dir}")
    return {"progress_percent": 100, "checkpoint_dir": result.checkpoint_dir}


@router.post("/api/recipe/recommend", response_model=RecipePresets)
async def recommend_recipe(req: TrainingStartRequest, state: State) -> dict[str, Any]:
    dataset = load_dataset(resolve_dataset(req.dataset_name))
    plan = state.planner().recommend(dataset, _training_request(req))
    return {"recipe": plan.recipe.model_dump(mode="json"), "presets": presets_payload(plan)}


@router.post("/api/training/start", status_code=202, response_model=JobStarted)
async def start_training(req: TrainingStartRequest, state: State) -> dict[str, Any]:
    dataset_path = resolve_dataset(req.dataset_name)
    cleaned = load_dataset(dataset_path)
    plan = state.planner().plan(cleaned, _training_request(req))
    recipe = plan.recipe
    if plan.model.gated:
        require_credential(HF_TOKEN)
    backend = state.build_training_backend()
    # Training waits until the backend is registered, so a cancel always finds it while the job runs.
    registered = threading.Event()

    def work(handle: JobHandle) -> dict[str, Any]:
        registered.wait()
        try:
            handle.log(f"Training {recipe.base_model} on {cleaned.total_records} records")
            handle.log(f"Objective: {recipe.objective.value.upper()} | Quantization: {recipe.quantization.value}")
            job = TrainingJob(id=handle.job_id, dataset_path=str(dataset_path), output_dir=req.output_dir,
                              recipe=recipe, status=JobStatus.RUNNING)
            return _train(handle, backend, job, cleaned)
        finally:
            state.backends.pop(handle.job_id, None)

    job_id = state.training.start({
        "dataset_name": dataset_path.name, "base_model": recipe.base_model,
        "objective": recipe.objective.value, "output_dir": req.output_dir, "progress_percent": 0,
        "current_step": 0, "total_steps": None, "current_loss": None, "loss_history": [],
        "checkpoint_dir": None, "error": None,
        "recipe": recipe.model_dump(mode="json"), "intensity": req.intensity,
    }, work)
    state.backends[job_id] = backend
    registered.set()
    return {"job_id": job_id, "status": "running"}


@router.get("/api/training", response_model=list[TrainingSummary])
async def list_training(state: State) -> list[dict[str, Any]]:
    return [summary(record) for record in state.store.list("training")]


@router.get("/api/training/status/{job_id}", response_model=TrainingRun)
async def get_training_status(job_id: str, state: State) -> dict[str, Any]:
    record = state.training.snapshot(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return record


@router.post("/api/training/cancel/{job_id}", response_model=TrainingCancelling)
async def cancel_training(job_id: str, state: State) -> dict[str, Any]:
    backend = state.backends.get(job_id)
    if backend is None:
        record = state.training.snapshot(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(status_code=409, detail=f"Job {job_id} is {record['status']}, so it cannot be cancelled")
    backend.cancel(job_id)
    # KeyError: the job ended between the lookup and this line, and its record is final.
    with contextlib.suppress(KeyError):
        JobHandle(state.training, job_id).log("Cancel requested. The trainer stops at its next step.")
    return {"status": "cancelling"}
