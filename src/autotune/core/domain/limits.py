"""Shared numeric limits and the VRAM validity rule."""
from __future__ import annotations

import math

GRPO_MAX_LEARNING_RATE: float = 2e-5
MAX_UPLOAD_BYTES: int = 100 * 1024 * 1024
MAX_FILE_BYTES: int = 20 * 1024 * 1024


def require_valid_vram(vram: float) -> float:
    """Return vram when it is finite and nonnegative, otherwise raise ValueError."""
    if not math.isfinite(vram) or vram < 0:
        raise ValueError("VRAM must be a finite nonnegative number")
    return vram
