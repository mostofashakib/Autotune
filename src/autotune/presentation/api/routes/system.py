"""What this server is: its GPU, the models that fit it, and which credentials are set."""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from autotune.core.domain.model_catalog import catalog_for_vram, default_model, fitting_quantization
from autotune.core.domain.models import TaskType
from autotune.presentation.api.schemas import CatalogModel, CredentialStatus, SystemStatus
from autotune.presentation.api.state import State
from autotune.settings import CREDENTIALS, ENV_FILE, credential_value, is_configured, ui_origin

router = APIRouter()


def recommended_quantization(vram: float) -> str:
    """Quantization of the default chat model for this VRAM, or "none" when nothing fits."""
    try:
        quantization = fitting_quantization(default_model(TaskType.CHAT, vram)[0], vram)
    except ValueError:
        return "none"
    return quantization.value if quantization else "none"


@router.get("/")
async def index() -> dict[str, str]:
    return {"name": "Autotune API", "ui": ui_origin(), "docs": "/docs"}


@router.get("/api/settings/credentials", response_model=list[CredentialStatus])
async def credentials() -> list[dict[str, Any]]:
    return [
        {
            "name": credential.env_var,
            "purpose": credential.purpose,
            "configured": is_configured(credential_value(credential)),
            "env_file": ENV_FILE.name,
        }
        for credential in CREDENTIALS
    ]


@router.get("/api/models", response_model=list[CatalogModel])
async def models(
    state: State, vram: Annotated[float | None, Query(ge=0, allow_inf_nan=False)] = None,
) -> list[dict]:
    return catalog_for_vram(state.build_hardware_probe().available_vram_gb() if vram is None else vram)


@router.get("/api/system/status", response_model=SystemStatus)
async def get_system_status(state: State) -> dict[str, Any]:
    vram = state.build_hardware_probe().available_vram_gb()
    has_cuda = vram > 0
    return {
        "vram_gb": round(vram, 2),
        "cuda_available": has_cuda,
        "device": "CUDA" if has_cuda else "CPU / no CUDA",
        "recommended_quantization": recommended_quantization(vram),
    }
