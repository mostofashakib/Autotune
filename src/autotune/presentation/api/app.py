"""FastAPI backend application for Autotune interactive dashboard."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from autotune.core.ports.hardware import HardwareProbe
from autotune.core.ports.trainer import TrainingBackend
from autotune.infrastructure.backends.trl_peft_backend import TrlPeftBackend
from autotune.infrastructure.hardware.cuda_probe import CudaProbe
from autotune.infrastructure.serving.vllm_provider import VLLMServingProvider
from autotune.infrastructure.storage.job_store import JobStore
from autotune.presentation.api.error_handlers import install_error_handlers
from autotune.presentation.api.jobs import JobRunner
from autotune.presentation.api.routes import benchmarks, cloud, datasets, serving, system, training
from autotune.presentation.api.state import AppState, ServingSlot
from autotune.settings import data_dir, export_configured_credentials, ui_origin


def build_training_backend() -> TrainingBackend:
    """The backend that trains for real. Its missing dependencies or GPU fail the job."""
    return TrlPeftBackend()


def build_hardware_probe() -> HardwareProbe:
    """The probe that reads the GPU this server trains on."""
    return CudaProbe()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    export_configured_credentials()
    store = JobStore(data_dir() / "autotune.db")
    for kind in ("training", "benchmark", "ingestion"):
        store.mark_interrupted(kind)

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
    for router in (system.router, datasets.router, training.router, benchmarks.router, cloud.router, serving.router):
        app.include_router(router)
    install_error_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[ui_origin()],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app
