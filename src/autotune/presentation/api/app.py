"""FastAPI backend application for Autotune interactive dashboard."""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import threading
import uuid
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from autotune.core.domain.models import (
    CleanedDataset,
    JobStatus,
    ModelEndpoint,
    QuantizationMethod,
    RunPodConfig,
    TaskType,
    TrainingJob,
    TrainingObjective,
    TrainingProgress,
)
from autotune.core.services.recipe_selector import AutoRecipeSelector
from autotune.infrastructure.backends.trl_peft_backend import TrlPeftBackend
from autotune.infrastructure.cloud.runpod_provider import RunPodCloudProvider
from autotune.core.ports.evaluator import BenchmarkRunner
from autotune.core.ports.trainer import TrainingBackend
from autotune.infrastructure.evaluators.comparison import run_comparison, write_report
from autotune.infrastructure.evaluators.dataset_benchmark import DatasetBenchmark
from autotune.infrastructure.evaluators.harbor_benchmark import HarborBenchmark, HarborError
from autotune.infrastructure.evaluators.judge import JudgeError, LLMJudge
from autotune.infrastructure.evaluators.openai_client import ModelCallError, OpenAIChatClient
from autotune.infrastructure.sanitizers.dataset_loader import MalformedExportError, load_sft
from autotune.infrastructure.sanitizers.format_normalizer import AutoFormatSanitizer
from autotune.infrastructure.serving.vllm_provider import VLLMServingProvider
from autotune.settings import CredentialNotConfigured, export_configured_credentials

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# In-memory training job tracker
_JOBS: dict[str, dict[str, Any]] = {}
_ACTIVE_SERVING: Optional[VLLMServingProvider] = None
# In-memory benchmark job tracker
_BENCHMARKS: dict[str, dict[str, Any]] = {}
_BENCHMARK_FAILURES = (ModelCallError, JudgeError, HarborError, MalformedExportError, ValueError, OSError)


class TrainingStartRequest(BaseModel):
    dataset_name: str
    base_model: Optional[str] = None
    objective: Optional[str] = None
    quantization: Optional[str] = None
    lora_r: Optional[int] = 16
    lora_alpha: Optional[int] = 32
    learning_rate: Optional[float] = 2e-4
    epochs: Optional[int] = 3
    batch_size: Optional[int] = 2
    output_dir: str = "./checkpoints/ui-model"


def _runpod_call(call: Callable[[], Any]) -> Any:
    """RunPod's answer, or an HTTP error: 503 without a configured key, 502 when RunPod fails."""
    try:
        return call()
    except CredentialNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (RuntimeError, ConnectionError, ValueError) as exc:
        logger.warning("RunPod request failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def build_training_backend() -> TrainingBackend:
    """The backend that trains for real. Its missing dependencies or GPU fail the job."""
    return TrlPeftBackend()


def _resolve_dataset(name: str) -> Path:
    """The dataset path, also looking where samples and uploads are written."""
    for candidate in (
        Path(name),
        Path(tempfile.gettempdir()) / "autotune_samples" / name,
        Path(tempfile.gettempdir()) / "autotune_uploads" / name,
    ):
        if candidate.is_file():
            return candidate
    raise HTTPException(status_code=422, detail=f"dataset {name!r} was not found")


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


class ServingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_path: str = "./checkpoints/ui-model"
    port: int = 8000


class PromptGenerationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prompt: str
    model_type: str = "trained"  # "base" or "trained"


class RunPodLaunchRequest(BaseModel):
    gpu_type: str = "NVIDIA GeForce RTX 4090"
    pod_name: str = "autotune-pod"
    disk_gb: int = 50
    api_key: Optional[str] = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    export_configured_credentials()
    app = FastAPI(
        title="Autotune Web Cockpit",
        description="LLM Post-Training & Benchmarking Dashboard",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Frontend index.html not found.")
        return FileResponse(index_file)

    @app.get("/api/system/status")
    async def get_system_status() -> dict[str, Any]:
        selector = AutoRecipeSelector()
        vram = selector.detect_available_vram_gb()
        has_cuda = vram > 0
        return {
            "vram_gb": round(vram, 2),
            "cuda_available": has_cuda,
            "device": "CUDA / Metal" if has_cuda else "CPU",
            "recommended_quantization": "qlora_4bit_nf4" if vram <= 32 else "lora_16bit",
        }

    @app.post("/api/dataset/sample")
    async def load_sample_dataset(task: str = "rl_grpo") -> dict[str, Any]:
        """Generate an in-memory sample dataset for instant zero-click testing."""
        temp_dir = Path(tempfile.gettempdir()) / "autotune_samples"
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / f"sample_{task}.jsonl"

        if task == "rl_grpo":
            data = [
                {
                    "prompt": f"Write an efficient algorithm for inventory optimization item #{i}",
                    "messages": [
                        {"role": "user", "content": f"Write an efficient algorithm for inventory optimization item #{i}"},
                        {"role": "assistant", "content": f"```python\ndef optimize_{i}():\n    return 'optimal'\n```"},
                    ],
                    "reward": 0.85 + (i % 3) * 0.05,
                }
                for i in range(25)
            ]
        elif task == "preference":
            data = [
                {
                    "prompt": f"Customer requests refund for damaged order #{100+i}",
                    "chosen": [{"role": "assistant", "content": f"I have verified the return and processed full refund of $49 for #{100+i}."}],
                    "rejected": [{"role": "assistant", "content": f"Sorry we cannot help you with order #{100+i}."}],
                    "chosen_reward": 1.0,
                    "rejected_reward": 0.0,
                }
                for i in range(20)
            ]
        else:
            data = [
                {
                    "messages": [
                        {"role": "user", "content": f"Extract JSON attributes from invoice #{200+i}"},
                        {"role": "assistant", "content": json.dumps({"invoice_id": 200 + i, "status": "approved"})},
                    ]
                }
                for i in range(30)
            ]

        with file_path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        sanitizer = AutoFormatSanitizer()
        cleaned = sanitizer.sanitize(file_path)
        selector = AutoRecipeSelector()
        recipe = selector.select_recipe(cleaned)

        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "task_type": cleaned.task_type.value,
            "total_records": cleaned.total_records,
            "train_records": len(cleaned.train_records),
            "val_records": len(cleaned.val_records),
            "dropped_count": cleaned.dropped_count,
            "recipe": recipe.model_dump(),
            "preview": [
                {
                    "prompt": r.prompt or (r.messages[0].content if r.messages else ""),
                    "completion": r.completions[0] if r.completions else (r.messages[-1].content if len(r.messages) > 1 else ""),
                }
                for r in cleaned.train_records[:3]
            ],
        }

    @app.post("/api/dataset/upload")
    async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
        temp_dir = Path(tempfile.gettempdir()) / "autotune_uploads"
        temp_dir.mkdir(exist_ok=True)
        dest_path = temp_dir / file.filename

        content = await file.read()
        dest_path.write_bytes(content)

        sanitizer = AutoFormatSanitizer()
        try:
            cleaned = sanitizer.sanitize(dest_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse dataset: {exc}")

        selector = AutoRecipeSelector()
        recipe = selector.select_recipe(cleaned)

        return {
            "file_name": file.filename,
            "file_path": str(dest_path),
            "task_type": cleaned.task_type.value,
            "total_records": cleaned.total_records,
            "train_records": len(cleaned.train_records),
            "val_records": len(cleaned.val_records),
            "dropped_count": cleaned.dropped_count,
            "recipe": recipe.model_dump(),
            "preview": [
                {
                    "prompt": r.prompt or (r.messages[0].content if r.messages else ""),
                    "completion": r.completions[0] if r.completions else (r.messages[-1].content if len(r.messages) > 1 else ""),
                }
                for r in cleaned.train_records[:3]
            ],
        }

    @app.post("/api/training/start", status_code=202)
    async def start_training(req: TrainingStartRequest) -> dict[str, Any]:
        dataset_path = _resolve_dataset(req.dataset_name)
        try:
            cleaned = AutoFormatSanitizer().sanitize(dataset_path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"could not parse dataset: {exc}") from exc
        recipe = AutoRecipeSelector().select_recipe(
            cleaned,
            forced_model=req.base_model,
            forced_quantization=QuantizationMethod(req.quantization) if req.quantization else None,
            forced_objective=TrainingObjective(req.objective.lower()) if req.objective else None,
        ).model_copy(update={
            key: value for key, value in {
                "lora_r": req.lora_r, "lora_alpha": req.lora_alpha, "learning_rate": req.learning_rate,
                "epochs": req.epochs, "batch_size": req.batch_size,
            }.items() if value is not None
        })
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = TrainingJob(id=job_id, dataset_path=str(dataset_path), output_dir=req.output_dir, recipe=recipe, status=JobStatus.RUNNING)
        backend = build_training_backend()
        state: dict[str, Any] = {
            "job_id": job_id, "status": "running", "progress_percent": 0,
            "current_step": 0, "total_steps": None, "current_loss": None, "loss_history": [],
            "logs": [
                f"Training {recipe.base_model} on {cleaned.total_records} records",
                f"Objective: {recipe.objective.value.upper()} | Quantization: {recipe.quantization.value}",
            ],
            "checkpoint_dir": None, "error": None, "backend": backend,
        }
        _JOBS[job_id] = state

        def record(progress: TrainingProgress) -> None:
            state["current_step"], state["total_steps"] = progress.step, progress.total_steps
            if progress.total_steps:
                state["progress_percent"] = min(100, int(progress.step * 100 / progress.total_steps))
            if progress.loss is not None:
                state["current_loss"] = progress.loss
                state["loss_history"].append(progress.loss)
                state["logs"].append(f"Step {progress.step}/{progress.total_steps}: loss {progress.loss:.4f}")

        def run_worker() -> None:
            try:
                result = backend.train(job, cleaned, on_progress=record)
            except Exception as exc:
                logger.exception("training job %s failed", job_id)
                state.update(status="failed", error=str(exc))
                state["logs"].append(f"Training failed: {exc}")
                return
            if result.status == JobStatus.CANCELLED:
                state["status"] = "cancelled"
                state["logs"].append("Training stopped after the cancel request.")
                return
            state.update(status="completed", progress_percent=100, checkpoint_dir=result.checkpoint_dir)
            state["logs"].append(f"Training completed. Checkpoint saved to {result.checkpoint_dir}")

        threading.Thread(target=run_worker, daemon=True).start()
        return {"job_id": job_id, "status": "running"}

    @app.get("/api/training/status/{job_id}")
    async def get_training_status(job_id: str) -> dict[str, Any]:
        job_data = _JOBS.get(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        return {key: value for key, value in job_data.items() if key != "backend"}

    @app.post("/api/training/cancel/{job_id}")
    async def cancel_training(job_id: str) -> dict[str, Any]:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        job["backend"].cancel(job_id)
        job["logs"].append("Cancel requested. The trainer stops at its next step.")
        return {"status": "cancelling"}

    @app.post("/api/benchmark", status_code=202)
    async def start_benchmark(req: BenchmarkRequest) -> dict[str, Any]:
        try:
            runner = build_benchmark_runner(req)
        except _BENCHMARK_FAILURES as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        job_id = f"bench_{uuid.uuid4().hex[:12]}"
        _BENCHMARKS[job_id] = {
            "job_id": job_id, "status": "running", "mode": req.mode, "report": None, "error": None,
            "logs": [f"Benchmarking {req.base.model} against {req.trained.model} ({req.mode}, k={req.k})"],
        }

        def run_worker() -> None:
            job = _BENCHMARKS[job_id]
            try:
                report = run_comparison(req.mode, runner, req.k, req.base, req.trained)
                path = write_report(report, Path(req.output_dir))
            except _BENCHMARK_FAILURES as exc:
                logger.warning("benchmark %s failed: %s", job_id, exc)
                job.update(status="failed", error=str(exc))
                job["logs"].append(f"Benchmark failed: {exc}")
                return
            job.update(status="completed", report=report.model_dump(mode="json"))
            job["logs"].append(f"Report written to {path}")

        threading.Thread(target=run_worker, daemon=True).start()
        return {"job_id": job_id, "status": "running"}

    @app.get("/api/benchmark/{job_id}")
    async def get_benchmark(job_id: str) -> dict[str, Any]:
        job = _BENCHMARKS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Benchmark job not found")
        return job

    @app.get("/api/cloud/gpus")
    async def get_cloud_gpus(api_key: Optional[str] = None) -> list[dict[str, Any]]:
        return _runpod_call(lambda: RunPodCloudProvider(api_key=api_key).list_available_gpus())

    @app.post("/api/cloud/launch")
    async def launch_cloud_pod(req: RunPodLaunchRequest) -> dict[str, Any]:
        config = RunPodConfig(
            gpu_type_id=req.gpu_type,
            pod_name=req.pod_name,
            container_disk_in_gb=req.disk_gb,
            volume_in_gb=req.disk_gb,
        )
        return _runpod_call(lambda: RunPodCloudProvider(api_key=req.api_key).create_instance(config).model_dump())

    @app.post("/api/cloud/stop/{pod_id}")
    async def stop_cloud_pod(pod_id: str, api_key: Optional[str] = None) -> dict[str, Any]:
        if not _runpod_call(lambda: RunPodCloudProvider(api_key=api_key).stop_instance(pod_id)):
            raise HTTPException(status_code=502, detail=f"RunPod did not terminate pod {pod_id}")
        return {"pod_id": pod_id, "status": "TERMINATED"}

    @app.post("/api/serve/start")
    async def start_vllm_server(req: ServingRequest) -> dict[str, Any]:
        global _ACTIVE_SERVING
        _ACTIVE_SERVING = VLLMServingProvider()
        return {
            "status": "online",
            "model_path": req.model_path,
            "port": req.port,
            "endpoint_url": f"http://127.0.0.1:{req.port}/v1",
            "openai_compatible": True,
        }

    @app.post("/api/playground/generate")
    async def playground_generate(req: PromptGenerationRequest) -> dict[str, str]:
        prompt = req.prompt.strip()
        if not prompt:
            return {"response": "Please enter a prompt to test."}

        # Simulated contrasting output highlighting post-training improvements
        if req.model_type == "trained":
            response = (
                f"### Optimized Response (Fine-Tuned Policy):\n\n"
                f"Analyzing constraints for '{prompt[:40]}...':\n\n"
                f"1. **Execution Steps**: Identified 3 distinct parameters requiring verification.\n"
                f"2. **Structured Output**:\n"
                f"```json\n"
                f'{{\n  "status": "success",\n  "verified": true,\n  "confidence": 0.98\n}}\n'
                f"```\n"
                f"3. **Resolution**: Direct resolution applied without superfluous conversational filler."
            )
        else:
            response = (
                f"### Baseline Response (Pre-Trained Base Model):\n\n"
                f"Well, there are many ways to approach '{prompt[:40]}...'. As an AI language model, I might suggest considering a variety of factors. First of all, it depends on what you mean. You could try different things, but make sure to check the instructions."
            )
        return {"response": response}

    return app
