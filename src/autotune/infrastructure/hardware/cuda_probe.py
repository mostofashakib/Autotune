"""HardwareProbe backed by torch.cuda."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CudaProbe:
    """Report free CUDA memory, or 0.0 when torch or a CUDA device is unavailable."""

    def available_vram_gb(self) -> float:
        try:
            import torch
            if not torch.cuda.is_available():
                logger.debug("No CUDA device is available")
                return 0.0
            free_bytes, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        except (ImportError, RuntimeError) as exc:
            logger.debug("Could not inspect GPU VRAM: %s", exc)
            return 0.0
        return free_bytes / (1024 ** 3)
