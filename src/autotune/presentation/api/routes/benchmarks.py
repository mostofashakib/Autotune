"""Benchmark a base model against its trained version, and list the comparisons."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from autotune.core.domain.models import ModelEndpoint
from autotune.core.ports.evaluator import BenchmarkRunner
from autotune.defaults import ANSWER_MAX_TOKENS, BENCHMARK_DIR, HARBOR_JOBS_DIR, PASS_K
from autotune.infrastructure.evaluators.comparison import run_comparison, write_report
from autotune.infrastructure.evaluators.dataset_benchmark import DatasetBenchmark
from autotune.infrastructure.evaluators.harbor_benchmark import HarborBenchmark
from autotune.infrastructure.evaluators.judge import LLMJudge
from autotune.infrastructure.evaluators.openai_client import OpenAIChatClient
from autotune.infrastructure.sanitizers.dataset_loader import load_sft
from autotune.presentation.api.jobs import JobHandle, summary
from autotune.presentation.api.schemas import BenchmarkRun, BenchmarkSummary, JobStarted
from autotune.presentation.api.state import State
from autotune.presentation.errors import EXPECTED_FAILURES

router = APIRouter()


class BenchmarkRequest(BaseModel):
    mode: Literal["dataset", "harbor"]
    # A held-out export for dataset mode, or a Harbor task directory for harbor mode.
    path: str = Field(min_length=1)
    base: ModelEndpoint
    trained: ModelEndpoint
    k: int = Field(default=PASS_K, ge=1, le=50)
    judge: Optional[ModelEndpoint] = None
    # Answer token budget for dataset mode. Reasoning models need room to think.
    max_tokens: int = Field(default=ANSWER_MAX_TOKENS, ge=1, le=131072)
    output_dir: str = BENCHMARK_DIR
    jobs_dir: str = HARBOR_JOBS_DIR

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


def _benchmark_summary(record: dict[str, Any]) -> dict[str, Any]:
    report = record.get("report")
    headline = None if report is None else {
        "base_pass_at_1": report["base"]["pass_at_1"],
        "trained_pass_at_1": report["trained"]["pass_at_1"],
        "delta_pass_at_1": report["delta_pass_at_1"],
    }
    return {**summary(record), "headline": headline}


def _compare(handle: JobHandle, req: BenchmarkRequest, runner: BenchmarkRunner) -> dict[str, Any]:
    handle.log(f"Benchmarking {req.base.model} against {req.trained.model} ({req.mode}, k={req.k})")
    try:
        report = run_comparison(req.mode, runner, req.k, req.base, req.trained)
        path = write_report(report, Path(req.output_dir))
    except EXPECTED_FAILURES as exc:
        handle.log(f"Benchmark failed: {exc}")
        raise
    handle.log(f"Report written to {path}")
    return {"report": report.model_dump(mode="json")}


@router.post("/api/benchmark", status_code=202, response_model=JobStarted)
async def start_benchmark(req: BenchmarkRequest, state: State) -> dict[str, Any]:
    runner = build_benchmark_runner(req)
    job_id = state.benchmarks.start({
        "mode": req.mode, "k": req.k, "path": req.path,
        "base_model": req.base.model, "trained_model": req.trained.model, "report": None, "error": None,
    }, lambda handle: _compare(handle, req, runner))
    return {"job_id": job_id, "status": "running"}


@router.get("/api/benchmark", response_model=list[BenchmarkSummary])
async def list_benchmarks(state: State) -> list[dict[str, Any]]:
    return [_benchmark_summary(record) for record in state.store.list("benchmark")]


@router.get("/api/benchmark/{job_id}", response_model=BenchmarkRun)
async def get_benchmark(job_id: str, state: State) -> dict[str, Any]:
    record = state.benchmarks.snapshot(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Benchmark job not found")
    return record
