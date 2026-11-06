"""Single pre-training flow shared by recommendations and training starts."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from autotune.core.domain.model_catalog import ModelSpec
from autotune.core.domain.models import CleanedDataset, QuantizationMethod, Recipe, TrainingObjective
from autotune.core.domain.training_settings import (
    TrainingIntensity, intensity_recipes, validate_rows_for, validate_training_recipe,
)
from autotune.core.ports.hardware import HardwareProbe
from autotune.core.services.recipe_selector import AutoRecipeSelector

TUNABLE_FIELDS = ("lora_r", "lora_alpha", "learning_rate", "epochs", "batch_size",
                  "gradient_accumulation_steps", "max_seq_length")


@dataclass(frozen=True)
class TrainingRequest:
    base_model: str | None = None
    objective: TrainingObjective | None = None
    quantization: QuantizationMethod | None = None
    intensity: TrainingIntensity = "balanced"
    overrides: Mapping[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingPlan:
    recipe: Recipe
    model: ModelSpec
    presets: dict[str, Recipe]
    vram_gb: float


class TrainingPlanner:
    """Select, adjust, and validate a training recipe against the probed hardware."""

    def __init__(self, probe: HardwareProbe, resolve_model: Callable[[str], ModelSpec]) -> None:
        self._probe = probe
        self._resolve_model = resolve_model

    def recommend(self, dataset: CleanedDataset, request: TrainingRequest) -> TrainingPlan:
        """Return the baseline recipe and presets. Skips memory validation, intensity, and overrides."""
        return self._select(dataset, request, self._probe.available_vram_gb())

    def plan(self, dataset: CleanedDataset, request: TrainingRequest) -> TrainingPlan:
        """Return a validated recipe with the request's intensity and overrides applied."""
        unknown = sorted(set(request.overrides) - set(TUNABLE_FIELDS))
        if unknown:
            raise ValueError(f"Unknown override: {', '.join(unknown)}. Tunable fields: {', '.join(TUNABLE_FIELDS)}")
        vram = self._probe.available_vram_gb()
        baseline = self._select(dataset, request, vram)
        selected = baseline.presets[request.intensity]
        recipe = Recipe.model_validate({**selected.model_dump(), **request.overrides})
        validate_training_recipe(recipe, baseline.model, vram)
        validate_rows_for(recipe.objective, dataset.train_records + dataset.val_records)
        return TrainingPlan(recipe=recipe, model=baseline.model, presets=baseline.presets, vram_gb=vram)

    def _select(self, dataset: CleanedDataset, request: TrainingRequest, vram: float) -> TrainingPlan:
        forced = self._resolve_model(request.base_model) if request.base_model else None
        recipe = AutoRecipeSelector().select_recipe(
            dataset, available_vram_gb=vram, forced_model=request.base_model,
            forced_quantization=request.quantization, forced_objective=request.objective,
            model_spec=forced, validate_memory=False)
        model = forced or self._resolve_model(recipe.base_model)
        return TrainingPlan(recipe=recipe, model=model, presets=intensity_recipes(recipe), vram_gb=vram)
