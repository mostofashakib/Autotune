"""The keywords the training backend passes to TRL must exist in the installed TRL release."""
import inspect

import pytest

trl = pytest.importorskip("trl")

pytestmark = pytest.mark.contract

COMMON = {"output_dir", "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps",
          "num_train_epochs", "gradient_checkpointing", "report_to"}
BACKEND_KEYWORDS = {
    "SFTConfig": COMMON | {"max_length", "warmup_ratio", "logging_steps", "save_strategy", "eval_strategy", "fp16", "bf16"},
    "DPOConfig": COMMON | {"max_length"},
    "GRPOConfig": COMMON | {"num_generations", "max_completion_length"},
}


def parameters(cls):
    return set(inspect.signature(cls.__init__).parameters)


def test_grpo_trainer_accepts_reward_functions():
    assert {"reward_funcs", "processing_class", "train_dataset"} <= parameters(trl.GRPOTrainer)


@pytest.mark.parametrize("config_name", sorted(BACKEND_KEYWORDS))
def test_config_accepts_every_backend_keyword(config_name):
    missing = BACKEND_KEYWORDS[config_name] - parameters(getattr(trl, config_name))
    assert not missing, f"{config_name} no longer accepts {sorted(missing)}"
