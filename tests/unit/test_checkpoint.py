"""PolicyCheckpoint save and load tests."""
from __future__ import annotations

import pytest
from autotune.infrastructure.storage.checkpoint import PolicyCheckpoint


def test_checkpoint_save_load_round_trip(tmp_path):
    ckpt = PolicyCheckpoint(
        objective="grpo",
        base_model="Qwen/Qwen2.5-3B",
        model_path=str(tmp_path / "autotune_policy"),
        num_examples=12,
        mean_reward=0.62,
    )
    ckpt.save(tmp_path)
    loaded = PolicyCheckpoint.load(tmp_path)
    assert loaded.objective == "grpo"
    assert loaded.num_examples == 12
    assert loaded.model_path == str(tmp_path / "autotune_policy")
    assert loaded.created_at  # stamped on save


def test_load_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PolicyCheckpoint.load(tmp_path)
