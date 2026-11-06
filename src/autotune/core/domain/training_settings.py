"""Training intensity and memory validation shared by API recommendations and runs."""
from __future__ import annotations

import math
from typing import Literal
from autotune.core.domain.limits import GRPO_MAX_LEARNING_RATE, require_valid_vram
from autotune.core.domain.model_catalog import ModelSpec, memory_baseline
from autotune.core.domain.models import DatasetRecord, Recipe, TrainingObjective
from autotune.core.domain.rewards import row_reward

TrainingIntensity = Literal["quick", "balanced", "thorough"]


def intensity_recipes(recipe: Recipe) -> dict[str, Recipe]:
    """Scale epochs/LR from the recommendation: quick .5/.5, balanced 1/1, thorough 1.5/.75."""
    result = {}
    for name, epoch_scale, rate_scale in (("quick", .5, .5), ("balanced", 1, 1), ("thorough", 1.5, .75)):
        rate = round(recipe.learning_rate * rate_scale, 12)
        if recipe.objective == TrainingObjective.GRPO:
            rate = min(rate, GRPO_MAX_LEARNING_RATE)
        result[name] = recipe.model_copy(update={"epochs": max(1, math.ceil(recipe.epochs * epoch_scale)), "learning_rate": rate})
    return result


def estimate_memory_gb(recipe: Recipe, model: ModelSpec) -> float:
    """Budget weights plus activations scaled by batch, sequence, and adapter rank.

    DPO carries extra chosen/rejected activations. GRPO budgets its generation
    group. Gradient accumulation does not multiply concurrent activation memory.
    Assumes gradient checkpointing, as enabled by the training backend.
    """
    baseline = memory_baseline(model.parameters_b, recipe.quantization)
    tokens_scale = recipe.batch_size * recipe.max_seq_length / 2048
    if recipe.objective == TrainingObjective.DPO:
        tokens_scale *= 2
    elif recipe.objective == TrainingObjective.GRPO:
        tokens_scale *= recipe.grpo_group_size
    activations = model.parameters_b * .25 * max(0, tokens_scale - 1)
    adapters = model.parameters_b * .1 * max(0, recipe.lora_r / 16 - 1)
    return round(baseline + activations + adapters, 2)


def validate_training_recipe(recipe: Recipe, model: ModelSpec, vram: float) -> None:
    """Reject invalid or oversized recipes with actionable messages before training."""
    positive = ("lora_r", "lora_alpha", "epochs", "batch_size", "gradient_accumulation_steps", "max_seq_length")
    for name in positive:
        if getattr(recipe, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(recipe.learning_rate) or not 0 < recipe.learning_rate <= .1:
        raise ValueError("learning_rate must be finite, positive, and at most 0.1")
    if recipe.objective == TrainingObjective.GRPO and recipe.learning_rate > GRPO_MAX_LEARNING_RATE:
        raise ValueError("GRPO learning_rate must be at most 0.00002")
    if recipe.max_seq_length > model.context_length:
        raise ValueError(f"max_seq_length exceeds {model.hf_id}'s context length {model.context_length}")
    require_valid_vram(vram)
    needed = estimate_memory_gb(recipe, model)
    # Inspection and CPU-only development remain available. The real backend
    # checks CUDA before any tokenizer/model download.
    if vram > 0 and needed > vram:
        raise ValueError(f"Estimated memory {needed:g}GB exceeds {vram:g}GB VRAM. Reduce batch size or max sequence length, use QLoRA, or choose a smaller model.")


def validate_rows_for(objective: TrainingObjective, records: list[DatasetRecord]) -> None:
    """Reject records the selected objective cannot train on."""
    if objective == TrainingObjective.GRPO:
        unrewarded = sum(1 for record in records if row_reward(record.metadata) is None)
        if unrewarded:
            raise ValueError(f"GRPO needs a numeric reward in every row. Rows without one: {unrewarded}. "
                             "Add total_reward or reward to each row, or select SFT.")
        return
    if objective != TrainingObjective.DPO:
        return
    for record in records:
        chosen, rejected = record.metadata.get("chosen"), record.metadata.get("rejected")
        if not record.prompt or not isinstance(chosen, str) or not isinstance(rejected, str) or not chosen or not rejected or chosen == rejected:
            raise ValueError("DPO requires a prompt and distinct chosen/rejected responses in every row. Upload only preference rows or select SFT.")
