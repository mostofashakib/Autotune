"""Hardware and dataset-aware automatic recipe selection for fine-tuning."""
from __future__ import annotations

import logging
import math
from autotune.core.domain.model_catalog import MODELS, ModelSpec, fitting_quantization, get_model, memory_baseline
from autotune.core.domain.models import CleanedDataset, QuantizationMethod, Recipe, TaskType, TrainingObjective

logger = logging.getLogger(__name__)


class AutoRecipeSelector:
    """Infers optimal base model, LoRA/QLoRA quantization, and hyperparameters.

    Ensures users with limited experience do not suffer CUDA Out-Of-Memory errors
    or catastrophic forgetting.
    """

    @staticmethod
    def detect_available_vram_gb() -> float:
        """Inspect CUDA devices to determine available VRAM in Gigabytes."""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_bytes = torch.cuda.get_device_properties(device).total_memory
                return total_bytes / (1024 ** 3)
            # Check Apple Silicon MPS
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Apple unified memory typically shared
                return 16.0
        except Exception as exc:
            logger.debug("Could not inspect GPU VRAM: %s", exc)

        return 0.0  # CPU fallback or unknown

    def select_recipe(
        self,
        dataset: CleanedDataset,
        forced_model: str | None = None,
        forced_quantization: QuantizationMethod | None = None,
        forced_objective: TrainingObjective | None = None,
        available_vram_gb: float | None = None,
        model_spec: ModelSpec | None = None,
    ) -> Recipe:
        """Derive an optimal training recipe based on dataset characteristics and hardware."""
        vram = available_vram_gb if available_vram_gb is not None else self.detect_available_vram_gb()
        total_samples = dataset.total_records

        # 1. Objective selection
        if forced_objective is not None:
            objective = forced_objective
        elif dataset.task_type == TaskType.RL_GRPO:
            objective = TrainingObjective.GRPO
        elif dataset.task_type == TaskType.PREFERENCE:
            objective = TrainingObjective.DPO
        else:
            objective = TrainingObjective.SFT

        if not math.isfinite(vram) or vram < 0:
            raise ValueError("VRAM must be a finite nonnegative number")
        if forced_model:
            model = get_model(forced_model) or model_spec
            if model is None or model.hf_id != forced_model:
                raise ValueError(f"Unknown parameter count for {forced_model}. Resolve its Hugging Face metadata before selecting a recipe.")
        else:
            candidates = [m for m in MODELS if not m.gated and
                          (dataset.task_type != TaskType.TOOL_USE or m.supports_tool_calling) and
                          (vram == 0 or fitting_quantization(m, vram) is not None)]
            if not candidates:
                raise ValueError("No catalog model fits this GPU. Use a GPU with at least 5GB VRAM for QLoRA.")
            # No GPU still permits inspection. Training's GPU check refuses execution.
            model = max(candidates, key=lambda m: m.parameter_count) if vram else get_model("Qwen/Qwen2.5-1.5B-Instruct")
        base_model = model.hf_id
        quantization = forced_quantization or (fitting_quantization(model, vram) if vram else QuantizationMethod.NONE)
        if quantization is None:
            raise ValueError(f"{base_model} needs ~{model.qlora_min_vram_gb:g}GB VRAM with QLoRA. Choose a smaller model or larger GPU.")
        if vram and memory_baseline(model.parameters_b, quantization) > vram:
            raise ValueError(f"{base_model} does not fit {vram:g}GB VRAM at {quantization.value}. Select QLoRA or a smaller model.")

        # 3. Hyperparameters based on dataset volume
        if total_samples < 500:
            # Few-shot fine-tuning: higher rank to capture nuances, lower LR to prevent collapse
            lora_r = 32
            lora_alpha = 64
            learning_rate = 1e-4
            epochs = 4
            batch_size = 2 if vram >= 16.0 else 1
            grad_accum = 8
        elif total_samples < 5000:
            # Medium dataset: standard stable defaults
            lora_r = 16
            lora_alpha = 32
            learning_rate = 2e-4
            epochs = 3
            batch_size = 4 if vram >= 24.0 else 2
            grad_accum = 4
        else:
            # Large dataset: lower rank for efficiency, fewer epochs to prevent overfitting
            lora_r = 16
            lora_alpha = 32
            learning_rate = 3e-4
            epochs = 2
            batch_size = 8 if vram >= 32.0 else 4
            grad_accum = 2

        if objective == TrainingObjective.GRPO:
            learning_rate = min(learning_rate, 2e-5)

        return Recipe(
            base_model=base_model,
            objective=objective,
            quantization=quantization,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            epochs=epochs,
            max_seq_length=2048,
            warmup_ratio=0.05,
            grpo_group_size=4,
            grpo_epsilon=0.2,
        )
