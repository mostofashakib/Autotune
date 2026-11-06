"""Launch, list, and stop RunPod GPU pods, keeping a record of each pod Autotune launched."""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from autotune.core.domain.models import RunPodConfig
from autotune.defaults import RUNPOD_DISK_GB, RUNPOD_GPU, RUNPOD_POD_NAME
from autotune.infrastructure.cloud.runpod_provider import RunPodCloudProvider
from autotune.presentation.api.schemas import Gpu, LaunchedPod, Pod, PodStopped
from autotune.presentation.api.state import State
from autotune.settings import CredentialNotConfigured

logger = logging.getLogger(__name__)

router = APIRouter()


class RunPodLaunchRequest(BaseModel):
    gpu_type: str = RUNPOD_GPU
    pod_name: str = RUNPOD_POD_NAME
    disk_gb: int = RUNPOD_DISK_GB
    api_key: Optional[str] = None


def _runpod_call(call: Callable[[], Any]) -> Any:
    """RunPod's answer, or an HTTP error: 503 without a configured key, 502 when RunPod fails."""
    try:
        return call()
    except CredentialNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (RuntimeError, ConnectionError, ValueError) as exc:
        logger.warning("RunPod request failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/api/cloud/gpus", response_model=list[Gpu])
async def get_cloud_gpus() -> list[dict[str, Any]]:
    return _runpod_call(lambda: RunPodCloudProvider().list_available_gpus())


@router.post("/api/cloud/launch", response_model=LaunchedPod)
async def launch_cloud_pod(req: RunPodLaunchRequest, state: State) -> dict[str, Any]:
    config = RunPodConfig.single_disk(req.gpu_type, req.pod_name, req.disk_gb)
    pod = _runpod_call(lambda: RunPodCloudProvider(api_key=req.api_key).create_instance(config).model_dump())
    state.store.save("pod", pod["pod_id"], {**pod, "launched_at": datetime.now(timezone.utc).isoformat()})
    return pod


@router.get("/api/cloud/pods", response_model=list[Pod])
def list_cloud_pods(state: State) -> list[dict[str, Any]]:
    """Pods launched from Autotune, with RunPod's live status, or the stored record and why it is stale."""
    pods = []
    for record in state.store.list("pod"):
        if record["status"] == "TERMINATED":
            pods.append({**record, "live_error": None})
            continue
        try:
            live = RunPodCloudProvider().get_instance_status(record["pod_id"]).model_dump()
        except (CredentialNotConfigured, RuntimeError, ConnectionError, ValueError) as exc:
            pods.append({**record, "live_error": str(exc)})
        else:
            pods.append({**record, **live, "live_error": None})
    return pods


@router.post("/api/cloud/stop/{pod_id}", response_model=PodStopped)
async def stop_cloud_pod(pod_id: str, state: State) -> dict[str, Any]:
    if not _runpod_call(lambda: RunPodCloudProvider().stop_instance(pod_id)):
        raise HTTPException(status_code=502, detail=f"RunPod did not terminate pod {pod_id}")
    record = state.store.get("pod", pod_id)
    if record is not None:
        state.store.save("pod", pod_id, {**record, "status": "TERMINATED"})
    return {"pod_id": pod_id, "status": "TERMINATED"}
