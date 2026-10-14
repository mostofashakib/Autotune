"""Reviewed instruct models and conservative single-GPU adapter-training estimates."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from autotune.core.domain.models import QuantizationMethod


@dataclass(frozen=True)
class ModelSpec:
    """Hub identity, license, capabilities, and baseline training memory estimates."""
    hf_id: str
    display_name: str
    parameter_count: int
    license: str
    gated: bool
    supports_tool_calling: bool
    context_length: int
    qlora_min_vram_gb: float
    lora_min_vram_gb: float

    @property
    def parameters_b(self) -> float:
        return self.parameter_count / 1e9


def memory_baseline(parameters_b: float, quantization: QuantizationMethod) -> float:
    """Estimated GiB for batch 1, 2048 tokens, rank 16, gradient checkpointing."""
    # Engineering estimates, not measured vendor minima. Components follow
    # https://huggingface.co/docs/transformers/model_memory_anatomy:
    # frozen weights, adapter gradients/optimizer, activations, CUDA workspace.
    # Budget 0.75 GiB/B for NF4 weights/scales (2 GiB/B for 16-bit, 4 for NONE),
    # 0.25 GiB/B for checkpointed activations/adapters, plus 4 GiB workspace.
    # Every catalog estimate below uses this same explicitly conservative rule.
    weight = 0.75 if quantization == QuantizationMethod.QLORA_4BIT_NF4 else (2 if quantization == QuantizationMethod.LORA_16BIT else 4)
    return math.ceil(parameters_b * (weight + 0.25) + 4)


def _model(hf_id: str, name: str, billions: float, license: str, gated: bool, tools: bool, context: int) -> ModelSpec:
    return ModelSpec(hf_id, name, int(billions * 1e9), license, gated, tools, context,
                     memory_baseline(billions, QuantizationMethod.QLORA_4BIT_NF4),
                     memory_baseline(billions, QuantizationMethod.LORA_16BIT))


# IDs and metadata checked against each https://huggingface.co/<hf_id> model
# card on 2026-10-14. Parameter counts are rounded published model sizes.
# These are open-weight models with differing licenses, not all OSI open source.
MODELS = (
    _model("Qwen/Qwen2.5-0.5B-Instruct", "Qwen 2.5 0.5B", 0.49, "apache-2.0", False, True, 32768),
    _model("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B", 1.24, "llama3.2", True, False, 131072),
    _model("Qwen/Qwen2.5-1.5B-Instruct", "Qwen 2.5 1.5B", 1.54, "apache-2.0", False, True, 32768),
    _model("Qwen/Qwen2.5-3B-Instruct", "Qwen 2.5 3B", 3.09, "qwen-research", False, True, 32768),
    _model("meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B", 3.21, "llama3.2", True, True, 131072),
    _model("microsoft/Phi-4-mini-instruct", "Phi 4 Mini", 3.8, "mit", False, True, 131072),
    _model("mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B v0.3", 7.25, "apache-2.0", False, True, 32768),
    _model("Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5 7B", 7.61, "apache-2.0", False, True, 32768),
    _model("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B", 8.03, "llama3.1", True, True, 131072),
    _model("google/gemma-2-9b-it", "Gemma 2 9B", 9.24, "gemma", True, False, 8192),
    _model("Qwen/Qwen2.5-14B-Instruct", "Qwen 2.5 14B", 14.7, "apache-2.0", False, True, 32768),
)


def get_model(hf_id: str) -> ModelSpec | None:
    """Look up a reviewed model without network access."""
    return next((model for model in MODELS if model.hf_id == hf_id), None)


def fitting_quantization(model: ModelSpec, vram: float) -> QuantizationMethod | None:
    """Prefer 16-bit LoRA if it fits, otherwise NF4, otherwise no fit."""
    if not math.isfinite(vram) or vram < 0:
        raise ValueError("VRAM must be a finite nonnegative number")
    if vram >= model.lora_min_vram_gb:
        return QuantizationMethod.LORA_16BIT
    if vram >= model.qlora_min_vram_gb:
        return QuantizationMethod.QLORA_4BIT_NF4
    return None


def catalog_for_vram(vram: float) -> list[dict]:
    """All entries, including those that need more memory than is available."""
    rows = []
    for model in MODELS:
        quantization = fitting_quantization(model, vram)
        rows.append({**asdict(model), "parameters_b": model.parameters_b, "fits": quantization is not None,
                     "quantization": quantization.value if quantization else None,
                     "reason": None if quantization else f"needs ~{model.qlora_min_vram_gb:g}GB with QLoRA"})
    return rows
