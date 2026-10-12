"""Trainer log entries become progress updates only when they carry a training loss."""
from __future__ import annotations

from autotune.core.domain.models import TrainingProgress
from autotune.infrastructure.backends.trl_peft_backend import progress_from_log


def test_a_loss_log_becomes_a_progress_update():
    assert progress_from_log(10, 40, {"loss": 1.5, "learning_rate": 2e-4}) == TrainingProgress(step=10, total_steps=40, loss=1.5)


def test_an_eval_or_summary_log_without_loss_is_not_progress():
    assert progress_from_log(40, 40, {"eval_loss": 0.9}) is None
    assert progress_from_log(40, 40, {"train_runtime": 12.0}) is None


def test_a_non_numeric_loss_is_not_progress():
    assert progress_from_log(5, 40, {"loss": "nan?"}) is None
