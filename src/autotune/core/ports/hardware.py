"""HardwareProbe protocol for inspecting the machine that will run training."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class HardwareProbe(Protocol):
    """Port for reading hardware capacity."""

    def available_vram_gb(self) -> float:
        """Return free GPU memory in GiB, or 0.0 when no usable GPU exists."""
        ...
