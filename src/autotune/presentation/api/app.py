"""FastAPI backend application for Autotune interactive dashboard."""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

from autotune.core.domain.models import (
    JobStatus,
    ModelEndpoint,
    QuantizationMethod,
    TrainingJob,
    TrainingObjective,
    TrainingProgress,
)
from autotune.core.domain.training_settings import TrainingIntensity
from autotune.core.services.training_planner import TUNABLE_FIELDS, TrainingPlanner, TrainingRequest
from autotune.infrastructure.backends.trl_peft_backend import TrlPeftBackend
from autotune.infrastructure.models.hub_metadata import resolve_model
from autotune.core.ports.hardware import HardwareProbe
from autotune.core.ports.evaluator import BenchmarkRunner
from autotune.core.ports.trainer import TrainingBackend
from autotune.infrastructure.hardware.cuda_probe import CudaProbe
from autotune.infrastructure.evaluators.comparison import run_comparison, write_report
from autotune.infrastructure.evaluators.dataset_benchmark import DatasetBenchmark
from autotune.infrastructure.evaluators.harbor_benchmark import HarborBenchmark
from autotune.infrastructure.evaluators.judge import LLMJudge
from autotune.infrastructure.evaluators.openai_client import OpenAIChatClient
from autotune.infrastructure.sanitizers.dataset_loader import load_sft
from autotune.infrastructure.serving.vllm_provider import VLLMServingProvider
from autotune.infrastructure.storage.job_store import JobStore
from autotune.presentation.api.dataset_ingestion import load_dataset
from autotune.presentation.api.error_handlers import install_error_handlers
from autotune.presentation.api.jobs import JobRunner
from autotune.presentation.api.routes import cloud, datasets, serving, system
from autotune.presentation.api.routes.datasets import presets_payload, resolve_dataset
from autotune.presentation.api.state import AppState, ServingSlot
from autotune.presentation.errors import EXPECTED_FAILURES
from autotune.settings import (
    HF_TOKEN,
    require_credential,
    data_dir,
    export_configured_credentials,
    ui_origin,
)

logger = logging.getLogger(__name__)

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
    output_dir: str = "./checkpoints/ui-model"

    @field_validator("objective", mode="before")
    @classmethod
    def normalize_objective(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value


# Live-only keys never reach the job store or the API.
_LIVE_KEYS = {"backend"}
# The run list leaves out what only the detail view needs.
_DETAIL_KEYS = {"logs", "loss_history", "report"}


def _public(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key not in _LIVE_KEYS}


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in _DETAIL_KEYS}


def _benchmark_summary(record: dict[str, Any]) -> dict[str, Any]:
    report = record.get("report")
    headline = None if report is None else {
        "base_pass_at_1": report["base"]["pass_at_1"],
        "trained_pass_at_1": report["trained"]["pass_at_1"],
        "delta_pass_at_1": report["delta_pass_at_1"],
    }
    return {**_summary(record), "headline": headline}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_training_backend() -> TrainingBackend:
    """The backend that trains for real. Its missing dependencies or GPU fail the job."""
    return TrlPeftBackend()


def build_hardware_probe() -> HardwareProbe:
    """The probe that reads the GPU this server trains on."""
    return CudaProbe()


def _training_planner() -> TrainingPlanner:
    """A planner on the probe `build_hardware_probe` returns right now."""
    return TrainingPlanner(build_hardware_probe(), resolve_model)


def _training_request(req: TrainingStartRequest) -> TrainingRequest:
    fields = req.model_dump()
    return TrainingRequest(
        base_model=req.base_model, objective=req.objective, quantization=req.quantization,
        intensity=req.intensity,
        overrides={name: fields[name] for name in TUNABLE_FIELDS if fields[name] is not None})


class BenchmarkRequest(BaseModel):
    mode: Literal["dataset", "harbor"]
    # A held-out export for dataset mode, or a Harbor task directory for harbor mode.
    path: str = Field(min_length=1)
    base: ModelEndpoint
    trained: ModelEndpoint
    k: int = Field(default=3, ge=1, le=50)
    judge: Optional[ModelEndpoint] = None
    # Answer token budget for dataset mode. Reasoning models need room to think.
    max_tokens: int = Field(default=4096, ge=1, le=131072)
    output_dir: str = "benchmark_results"
    jobs_dir: str = "benchmark_results/harbor_jobs"

    @model_validator(mode="after")
    def _judge_needs_dataset_mode(self) -> "BenchmarkRequest":
        if self.judge is not None and self.mode != "dataset":
            raise ValueError("an LLM judge only applies to dataset mode")
        return self


def build_benchmark_runner(req: BenchmarkRequest) -> BenchmarkRunner:
    """The runner for `req`. Raises when its data or tasks cannot be used."""
    if req.mode == "harbor":
        return HarborBenchmark(Path(req.path), req.k, Path(req.jobs_dir))
    client = OpenAIChatClient()
    judge = LLMJudge(client, req.judge) if req.judge else None
    return DatasetBenchmark(load_sft(Path(req.path)), client, req.k, judge=judge, max_tokens=req.max_tokens)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    export_configured_credentials()
    store = JobStore(data_dir() / "autotune.db")
    for kind in ("training", "benchmark", "ingestion"):
        store.mark_interrupted(kind)
    # Live runs started by this server. Finished and earlier runs are read from the store.
    jobs: dict[str, dict[str, Any]] = {}
    benchmarks: dict[str, dict[str, Any]] = {}
    # Workers and requests both change a run's state. Changes and saves take turns.
    state_lock = threading.Lock()

    def save_training(state: dict[str, Any]) -> None:
        with state_lock:
            store.save("training", state["job_id"], _public(state))

    def save_benchmark(job: dict[str, Any]) -> None:
        store.save("benchmark", job["job_id"], job)

    app = FastAPI(
        title="Autotune Web Cockpit",
        description="LLM Post-Training & Benchmarking Dashboard",
        version="0.1.0",
    )
    # The lambdas read this module's factories when called, so tests can replace them after create_app.
    app.state.autotune = AppState(
        store=store,
        ingestion=JobRunner(store, "ingestion", id_prefix="ingest"),
        training=JobRunner(store, "training", id_prefix="job"),
        benchmarks=JobRunner(store, "benchmark", id_prefix="bench"),
        serving=ServingSlot(lambda: VLLMServingProvider()),
        build_training_backend=lambda: build_training_backend(),
        build_hardware_probe=lambda: build_hardware_probe(),
    )
    for router in (system.router, datasets.router, cloud.router, serving.router):
        app.include_router(router)
    install_error_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[ui_origin()],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/recipe/recommend")
    async def recommend_recipe(req: TrainingStartRequest) -> dict[str, Any]:
        dataset = load_dataset(resolve_dataset(req.dataset_name))
        plan = _training_planner().recommend(dataset, _training_request(req))
        return {"recipe": plan.recipe.model_dump(mode="json"), "presets": presets_payload(plan)}

    @app.post("/api/training/start", status_code=202)
    async def start_training(req: TrainingStartRequest) -> dict[str, Any]:
        dataset_path = resolve_dataset(req.dataset_name)
        cleaned = load_dataset(dataset_path)
        plan = _training_planner().plan(cleaned, _training_request(req))
        recipe = plan.recipe
        if plan.model.gated:
            require_credential(HF_TOKEN)
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = TrainingJob(id=job_id, dataset_path=str(dataset_path), output_dir=req.output_dir, recipe=recipe, status=JobStatus.RUNNING)
        backend = build_training_backend()
        state: dict[str, Any] = {
            "job_id": job_id, "status": "running", "created_at": _now(),
            "dataset_name": dataset_path.name, "base_model": recipe.base_model,
            "objective": recipe.objective.value, "output_dir": req.output_dir, "progress_percent": 0,
            "current_step": 0, "total_steps": None, "current_loss": None, "loss_history": [],
            "logs": [
                f"Training {recipe.base_model} on {cleaned.total_records} records",
                f"Objective: {recipe.objective.value.upper()} | Quantization: {recipe.quantization.value}",
            ],
            "checkpoint_dir": None, "error": None, "backend": backend,
            "recipe": recipe.model_dump(mode="json"), "intensity": req.intensity,
        }
        jobs[job_id] = state
        save_training(state)

        def record(progress: TrainingProgress) -> None:
            with state_lock:
                state["current_step"], state["total_steps"] = progress.step, progress.total_steps
                if progress.total_steps:
                    state["progress_percent"] = min(100, int(progress.step * 100 / progress.total_steps))
                if progress.loss is not None:
                    state["current_loss"] = progress.loss
                    state["loss_history"].append(progress.loss)
                    state["logs"].append(f"Step {progress.step}/{progress.total_steps}: loss {progress.loss:.4f}")
            save_training(state)

        def run_worker() -> None:
            try:
                result = backend.train(job, cleaned, on_progress=record)
            except Exception as exc:
                logger.exception("training job %s failed", job_id)
                with state_lock:
                    state.update(status="failed", error=str(exc))
                    state["logs"].append(f"Training failed: {exc}")
            else:
                with state_lock:
                    if result.status == JobStatus.CANCELLED:
                        state["status"] = "cancelled"
                        state["logs"].append("Training stopped after the cancel request.")
                    else:
                        state.update(status="completed", progress_percent=100, checkpoint_dir=result.checkpoint_dir)
                        state["logs"].append(f"Training completed. Checkpoint saved to {result.checkpoint_dir}")
            save_training(state)

        threading.Thread(target=run_worker, daemon=True).start()
        return {"job_id": job_id, "status": "running"}

    @app.get("/api/training")
    async def list_training() -> list[dict[str, Any]]:
        return [_summary(record) for record in store.list("training")]

    @app.get("/api/training/status/{job_id}")
    async def get_training_status(job_id: str) -> dict[str, Any]:
        if job_id in jobs:
            with state_lock:
                return json.loads(json.dumps(_public(jobs[job_id])))
        record = store.get("training", job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return record

    @app.post("/api/training/cancel/{job_id}")
    async def cancel_training(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            record = store.get("training", job_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Job not found")
            raise HTTPException(status_code=409, detail=f"Job {job_id} is {record['status']}, so it cannot be cancelled")
        job["backend"].cancel(job_id)
        with state_lock:
            job["logs"].append("Cancel requested. The trainer stops at its next step.")
        save_training(job)
        return {"status": "cancelling"}

    @app.post("/api/benchmark", status_code=202)
    async def start_benchmark(req: BenchmarkRequest) -> dict[str, Any]:
        runner = build_benchmark_runner(req)
        job_id = f"bench_{uuid.uuid4().hex[:12]}"
        job: dict[str, Any] = {
            "job_id": job_id, "status": "running", "created_at": _now(), "mode": req.mode, "k": req.k,
            "path": req.path, "base_model": req.base.model, "trained_model": req.trained.model,
            "report": None, "error": None,
            "logs": [f"Benchmarking {req.base.model} against {req.trained.model} ({req.mode}, k={req.k})"],
        }
        benchmarks[job_id] = job
        save_benchmark(job)

        def run_worker() -> None:
            try:
                report = run_comparison(req.mode, runner, req.k, req.base, req.trained)
                path = write_report(report, Path(req.output_dir))
            except EXPECTED_FAILURES as exc:
                logger.warning("benchmark %s failed: %s", job_id, exc)
                job.update(status="failed", error=str(exc))
                job["logs"].append(f"Benchmark failed: {exc}")
            else:
                job.update(status="completed", report=report.model_dump(mode="json"))
                job["logs"].append(f"Report written to {path}")
            save_benchmark(job)

        threading.Thread(target=run_worker, daemon=True).start()
        return {"job_id": job_id, "status": "running"}

    @app.get("/api/benchmark")
    async def list_benchmarks() -> list[dict[str, Any]]:
        return [_benchmark_summary(record) for record in store.list("benchmark")]

    @app.get("/api/benchmark/{job_id}")
    async def get_benchmark(job_id: str) -> dict[str, Any]:
        job = benchmarks.get(job_id) or store.get("benchmark", job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Benchmark job not found")
        return job

    return app
