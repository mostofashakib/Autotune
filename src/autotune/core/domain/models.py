"""Domain models and value objects for Autotune."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class TaskType(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    PREFERENCE = "preference"
    TOOL_USE = "tool_use"
    RL_GRPO = "rl_grpo"


class TrainingObjective(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"


class QuantizationMethod(str, Enum):
    NONE = "none"
    LORA_16BIT = "lora_16bit"
    QLORA_4BIT_NF4 = "qlora_4bit_nf4"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class DatasetRecord(BaseModel):
    id: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    prompt: str = ""
    completions: list[str] = Field(default_factory=list)
    rewards: list[float] = Field(default_factory=list)


class CleanedDataset(BaseModel):
    task_type: TaskType
    total_records: int
    train_records: list[DatasetRecord]
    val_records: list[DatasetRecord]
    dropped_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class Recipe(BaseModel):
    """Execution configuration automatically selected or customized."""
    base_model: str
    objective: TrainingObjective = TrainingObjective.SFT
    quantization: QuantizationMethod = QuantizationMethod.QLORA_4BIT_NF4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    epochs: int = 3
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    # GRPO Specific Settings
    grpo_group_size: int = 4
    grpo_epsilon: float = 0.2
    grpo_beta: float = 0.04


class ServingEndpoint(BaseModel):
    """vLLM or inference server endpoint metadata."""
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    base_url: str
    port: int = 8000
    backend: str = "vllm"
    is_ready: bool = False


class RunPodConfig(BaseModel):
    """Configuration for provisioning RunPod GPU instances."""
    api_key: str = ""
    gpu_type_id: str = "NVIDIA GeForce RTX 4090"
    cloud_type: str = "COMMUNITY"
    container_disk_in_gb: int = 50
    volume_in_gb: int = 50
    pod_name: str = "autotune-trainer"
    image_name: str = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"


class RunPodInstance(BaseModel):
    """Provisioned RunPod GPU pod instance metadata."""
    pod_id: str
    name: str
    status: str
    gpu_name: str
    cost_per_hr: float = 0.0
    ssh_ip: str | None = None
    ssh_port: int | None = None


class TrainingJob(BaseModel):
    id: str
    dataset_path: str
    output_dir: str
    recipe: Recipe
    status: JobStatus = JobStatus.PENDING
    error: str | None = None


class CheckpointResult(BaseModel):
    job_id: str
    checkpoint_dir: str
    final_loss: float = 0.0
    total_steps: int = 0
    duration_seconds: float = 0.0
    status: JobStatus = JobStatus.COMPLETED


class EvalMetric(BaseModel):
    task_name: str
    base_score: float
    trained_score: float
    delta: float
    passed: bool


class EvaluationReport(BaseModel):
    job_id: str
    base_model: str
    checkpoint_dir: str
    pass_at_1: float
    pass_at_3: float
    improvement_percentage: float
    metrics: list[EvalMetric] = Field(default_factory=list)
    summary: str = ""


RolloutOutcome = Literal["success", "failure", "partial_success", "edge_case", "truncated"]


class RolloutRecord(BaseModel):
    """Graded episode rollout record for RL and advantage estimation."""
    episode_id: str
    env_name: str = ""
    task_name: str = ""
    prompt: str = ""
    completion: str = ""
    seed: int | None = None
    total_reward: float = 0.0
    per_step_rewards: list[float] = Field(default_factory=list)
    passed: bool = False
    outcome: RolloutOutcome = "failure"
    steps: int = 0
    terminated: bool = False
    truncated: bool = False
    invalid_actions: int = 0
    error: str | None = None
    behavior_model: str = ""
    termination_reason: str = "unknown"
    verification_results: list[dict[str, Any]] = Field(default_factory=list)
    reward_breakdown: dict[str, Any] = Field(default_factory=dict)


class PreferenceRecord(BaseModel):
    """Chosen vs rejected trajectory pair for DPO alignment."""
    task: str = ""
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    chosen_passed: bool = True
    rejected_passed: bool = False
    env_name: str = ""


class SFTRecord(BaseModel):
    """Supervised fine-tuning demonstration record."""
    prompt: str
    completion: str
    env_name: str = ""
    reward: float = 1.0
    passed: bool = True
    behavior_model: str = ""


class EnvQualityMetrics(BaseModel):
    """Environment quality diagnostics."""
    env_name: str
    state_coverage_score: float
    reward_density: float
    dead_end_rate: float
    action_diversity: float
    num_episodes: int
    num_steps: int

