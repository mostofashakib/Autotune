"""Backend training engines and reward mapping infrastructure package for Autotune."""
from autotune.infrastructure.backends.mock_backend import MockTrainingBackend
from autotune.infrastructure.backends.reward_mapping import (
    DPOExample,
    GRPOExample,
    SFTExample,
    dpo_examples,
    grpo_advantages,
    sft_examples,
)
from autotune.infrastructure.backends.trl_peft_backend import TrlPeftBackend

__all__ = [
    "DPOExample",
    "GRPOExample",
    "MockTrainingBackend",
    "SFTExample",
    "TrlPeftBackend",
    "dpo_examples",
    "grpo_advantages",
    "sft_examples",
]
