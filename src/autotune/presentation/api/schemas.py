"""What the API answers. The dashboard's TypeScript types are generated from these models.

Job records keep every field the runner stores, so a model that lists fewer fields would drop
them from the JSON. Each model here names every field its route returns.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from autotune.core.domain.models import BenchmarkMode, ComparisonReport, Recipe
from autotune.core.domain.training_settings import TrainingIntensity


class SystemStatus(BaseModel):
    vram_gb: float
    cuda_available: bool
    device: str
    recommended_quantization: str


class CredentialStatus(BaseModel):
    name: str
    purpose: str
    configured: bool
    env_file: str


class CatalogModel(BaseModel):
    hf_id: str
    display_name: str
    fits: bool
    quantization: Literal["none", "lora_16bit", "qlora_4bit_nf4"] | None
    reason: str | None
    parameters_b: float
    parameter_count: int
    context_length: int
    license: str
    gated: bool
    supports_tool_calling: bool
    qlora_min_vram_gb: float
    lora_min_vram_gb: float


class Presets(BaseModel):
    quick: Recipe
    balanced: Recipe
    thorough: Recipe


class RecipePresets(BaseModel):
    recipe: Recipe
    presets: Presets


class PreviewRow(BaseModel):
    prompt: str
    completion: str


class UploadedDataset(BaseModel):
    file_name: str
    file_path: str
    task_type: str
    total_records: int
    train_records: int
    val_records: int
    dropped_count: int
    warnings: list[str]
    recipe: Recipe
    progress: dict[str, int]
    presets: Presets
    preview: list[PreviewRow]


class IngestionAccepted(BaseModel):
    job_id: str


class IngestionJob(BaseModel):
    """An upload being inspected. `result` appears once it finishes."""
    job_id: str
    status: str
    created_at: str
    logs: list[str]
    dataset_name: str
    progress: dict[str, int]
    error: str | None
    result: UploadedDataset | None = None


class JobStarted(BaseModel):
    job_id: str
    status: Literal["running"]


class TrainingSummary(BaseModel):
    job_id: str
    status: str
    created_at: str
    dataset_name: str | None = None
    base_model: str | None = None
    objective: str | None = None
    output_dir: str | None = None
    progress_percent: int
    current_step: int
    total_steps: int | None
    current_loss: float | None
    checkpoint_dir: str | None
    error: str | None
    recipe: Recipe | None = None
    intensity: TrainingIntensity | None = None


class TrainingRun(TrainingSummary):
    logs: list[str]
    loss_history: list[float]


class TrainingCancelling(BaseModel):
    status: Literal["cancelling"]


class Headline(BaseModel):
    base_pass_at_1: float
    trained_pass_at_1: float
    delta_pass_at_1: float


class BenchmarkJob(BaseModel):
    job_id: str
    status: str
    created_at: str
    mode: BenchmarkMode
    k: int
    path: str
    base_model: str
    trained_model: str
    error: str | None


class BenchmarkRun(BenchmarkJob):
    logs: list[str]
    report: ComparisonReport | None


class BenchmarkSummary(BenchmarkJob):
    headline: Headline | None


class Gpu(BaseModel):
    id: str
    name: str
    memory_gb: float
    community_price: float


class LaunchedPod(BaseModel):
    pod_id: str
    name: str
    status: str
    gpu_name: str
    cost_per_hr: float
    ssh_ip: str | None
    ssh_port: int | None


class Pod(LaunchedPod):
    launched_at: str
    live_error: str | None


class PodStopped(BaseModel):
    pod_id: str
    status: Literal["TERMINATED"]


class ServingStarted(BaseModel):
    status: Literal["starting"]
    model_path: str
    port: int
    endpoint_url: str


class ServingStopped(BaseModel):
    status: Literal["stopped"]


class PlaygroundAnswer(BaseModel):
    response: str

