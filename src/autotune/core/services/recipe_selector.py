"""Hardware and dataset-aware automatic recipe selection for fine-tuning."""
from __future__ import annotations

import math
from autotune.core.domain.limits import GRPO_MAX_LEARNING_RATE, require_valid_vram
from autotune.core.domain.model_catalog import ModelSpec, default_model, fitting_quantization, get_model, memory_baseline
from autotune.core.domain.training_settings import estimate_memory_gb, validate_training_recipe
from autotune.core.domain.models import CleanedDataset, QuantizationMethod, Recipe, TaskType, TrainingObjective


def choose_objective(task_type: TaskType, forced: TrainingObjective | None) -> TrainingObjective:
    """Use the forced objective, otherwise the one the task type implies."""
    if forced is not None:
        return forced
    if task_type == TaskType.RL_GRPO:
        return TrainingObjective.GRPO
    if task_type == TaskType.PREFERENCE:
        return TrainingObjective.DPO
    return TrainingObjective.SFT


def base_hyperparameters(total_samples: int, vram: float, objective: TrainingObjective) -> dict[str, float | int]:
    """Scale LoRA rank, learning rate, epochs, and batch size to dataset volume and VRAM."""
    if total_samples < 500:
        # Few-shot fine-tuning: higher rank to capture nuances, lower LR to prevent collapse
        params: dict[str, float | int] = {"lora_r": 32, "lora_alpha": 64, "learning_rate": 1e-4, "epochs": 4,
                                          "batch_size": 2 if vram >= 16.0 else 1, "gradient_accumulation_steps": 8}
    elif total_samples < 5000:
        # Medium dataset: standard stable defaults
        params = {"lora_r": 16, "lora_alpha": 32, "learning_rate": 2e-4, "epochs": 3,
                  "batch_size": 4 if vram >= 24.0 else 2, "gradient_accumulation_steps": 4}
    else:
        # Large dataset: lower rank for efficiency, fewer epochs to prevent overfitting
        params = {"lora_r": 16, "lora_alpha": 32, "learning_rate": 3e-4, "epochs": 2,
                  "batch_size": 8 if vram >= 32.0 else 4, "gradient_accumulation_steps": 2}
    if objective == TrainingObjective.GRPO:
        params["learning_rate"] = min(params["learning_rate"], GRPO_MAX_LEARNING_RATE)
    return params


def _resolve_model(dataset: CleanedDataset, vram: float, forced_model: str | None,
                   model_spec: ModelSpec | None) -> tuple[ModelSpec, list[ModelSpec]]:
    if not forced_model:
        return default_model(dataset.task_type, vram)
    model = get_model(forced_model) or model_spec
    if model is None or model.hf_id != forced_model:
        raise ValueError(f"Unknown parameter count for {forced_model}. Resolve its Hugging Face metadata before selecting a recipe.")
    return model, []


def _resolve_quantization(model: ModelSpec, vram: float, forced: QuantizationMethod | None) -> QuantizationMethod:
    quantization = forced or (fitting_quantization(model, vram) if vram else QuantizationMethod.NONE)
    if quantization is None:
        raise ValueError(f"{model.hf_id} needs ~{model.qlora_min_vram_gb:g}GB VRAM with QLoRA. Choose a smaller model or larger GPU.")
    if vram and memory_baseline(model.parameters_b, quantization) > vram:
        raise ValueError(f"{model.hf_id} does not fit {vram:g}GB VRAM at {quantization.value}. Select QLoRA or a smaller model.")
    return quantization


def _shrink_batch(recipe: Recipe, model: ModelSpec, vram: float) -> Recipe:
    """Lower the batch size until memory fits, keeping the effective batch size."""
    original_batch = recipe.batch_size
    while recipe.batch_size > 1 and estimate_memory_gb(recipe, model) > vram:
        recipe = recipe.model_copy(update={"batch_size": recipe.batch_size - 1})
    accumulation = math.ceil(recipe.gradient_accumulation_steps * original_batch / recipe.batch_size)
    return recipe.model_copy(update={"gradient_accumulation_steps": accumulation})


def _fall_back_to_smaller_model(recipe: Recipe, model: ModelSpec, candidates: list[ModelSpec], vram: float,
                                forced_quantization: QuantizationMethod | None) -> tuple[Recipe, ModelSpec]:
    smaller = [m for m in candidates if m.parameter_count < model.parameter_count]
    for candidate in sorted(smaller, key=lambda m: m.parameter_count, reverse=True):
        trial = recipe.model_copy(update={"base_model": candidate.hf_id,
                                          "quantization": forced_quantization or fitting_quantization(candidate, vram)})
        if estimate_memory_gb(trial, candidate) <= vram:
            return trial, candidate
    return recipe, model


def fit_to_memory(recipe: Recipe, model: ModelSpec, candidates: list[ModelSpec], vram: float, *,
                  forced_quantization: QuantizationMethod | None,
                  forced_model: str | None) -> tuple[Recipe, ModelSpec]:
    """Shrink batch size, then quantize, then pick a smaller model until the estimate fits VRAM."""
    if not vram:
        return recipe, model
    recipe = _shrink_batch(recipe, model, vram)
    if not forced_quantization and estimate_memory_gb(recipe, model) > vram:
        recipe = recipe.model_copy(update={"quantization": QuantizationMethod.QLORA_4BIT_NF4})
    if not forced_model and estimate_memory_gb(recipe, model) > vram:
        recipe, model = _fall_back_to_smaller_model(recipe, model, candidates, vram, forced_quantization)
    return recipe, model


class AutoRecipeSelector:
    """Select a model and data-driven hyperparameters within an estimated memory budget."""

    def select_recipe(
        self,
        dataset: CleanedDataset,
        *,
        available_vram_gb: float,
        forced_model: str | None = None,
        forced_quantization: QuantizationMethod | None = None,
        forced_objective: TrainingObjective | None = None,
        model_spec: ModelSpec | None = None,
        validate_memory: bool = True,
    ) -> Recipe:
        """Derive an optimal training recipe based on dataset characteristics and hardware."""
        vram = available_vram_gb
        objective = choose_objective(dataset.task_type, forced_objective)
        require_valid_vram(vram)
        model, candidates = _resolve_model(dataset, vram, forced_model, model_spec)
        quantization = _resolve_quantization(model, vram, forced_quantization)
        recipe = Recipe(base_model=model.hf_id, objective=objective, quantization=quantization,
                        **base_hyperparameters(dataset.total_records, vram, objective))
        recipe, model = fit_to_memory(recipe, model, candidates, vram,
                                      forced_quantization=forced_quantization, forced_model=forced_model)
        if vram and validate_memory:
            validate_training_recipe(recipe, model, vram)
        return recipe
