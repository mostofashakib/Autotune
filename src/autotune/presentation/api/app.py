"""FastAPI backend application for Autotune interactive dashboard."""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from autotune.core.domain.models import (
    ModelEndpoint,
)
from autotune.infrastructure.backends.trl_peft_backend import TrlPeftBackend
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
from autotune.presentation.api.error_handlers import install_error_handlers
from autotune.presentation.api.jobs import JobRunner, summary
from autotune.presentation.api.routes import cloud, datasets, serving, system, training
from autotune.presentation.api.state import AppState, ServingSlot
from autotune.presentation.errors import EXPECTED_FAILURES
from autotune.settings import (
    data_dir,
    export_configured_credentials,
    ui_origin,
)

logger = logging.getLogger(__name__)

def _benchmark_summary(record: dict[str, Any]) -> dict[str, Any]:
    report = record.get("report")
    headline = None if report is None else {
        "base_pass_at_1": report["base"]["pass_at_1"],
        "trained_pass_at_1": report["trained"]["pass_at_1"],
        "delta_pass_at_1": report["delta_pass_at_1"],
    }
    return {**summary(record), "headline": headline}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_training_backend() -> TrainingBackend:
    """The backend that trains for real. Its missing dependencies or GPU fail the job."""
    return TrlPeftBackend()


def build_hardware_probe() -> HardwareProbe:
    """The probe that reads the GPU this server trains on."""
    return CudaProbe()


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
    benchmarks: dict[str, dict[str, Any]] = {}

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
    for router in (system.router, datasets.router, training.router, cloud.router, serving.router):
        app.include_router(router)
    install_error_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[ui_origin()],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
