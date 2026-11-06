"""Upload a dataset, inspect it into a recommended recipe, and find uploaded datasets by name."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from autotune.core.domain.limits import MAX_UPLOAD_BYTES
from autotune.core.services.training_planner import TrainingPlan, TrainingPlanner, TrainingRequest
from autotune.presentation.api.dataset_ingestion import load_dataset
from autotune.presentation.api.jobs import JobHandle
from autotune.presentation.api.schemas import IngestionAccepted, IngestionJob, UploadedDataset
from autotune.presentation.api.state import State
from autotune.settings import data_dir

router = APIRouter()


def uploads_dir() -> Path:
    return data_dir() / "uploads"


def resolve_dataset(name: str) -> Path:
    """The dataset path, also looking in the uploads directory."""
    for candidate in (Path(name), uploads_dir() / name):
        if candidate.is_file():
            return candidate
    raise HTTPException(status_code=422, detail=f"dataset {name!r} was not found")


def presets_payload(plan: TrainingPlan) -> dict[str, dict]:
    return {name: preset.model_dump(mode="json") for name, preset in plan.presets.items()}


def _upload_name(filename: str | None) -> str:
    """The upload's file name, refusing anything that is a path rather than a bare name."""
    name = (filename or "").strip()
    if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
        raise HTTPException(status_code=400, detail=f"Upload name {filename!r} must be a plain file name")
    return name


def inspect_upload(
    dest_path: Path, planner: TrainingPlanner, on_progress: Callable[[dict[str, int]], None],
) -> dict[str, Any]:
    """What the upload holds and the recipe it calls for. A failed inspection deletes the upload."""
    last: dict[str, dict[str, int]] = {"progress": {}}

    def progress(update: dict[str, int]) -> None:
        last["progress"] = update
        on_progress(update)

    try:
        cleaned = load_dataset(dest_path, progress)
        plan = planner.recommend(cleaned, TrainingRequest())
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise
    return {
        "file_name": dest_path.name, "file_path": str(dest_path),
        "task_type": cleaned.task_type.value, "total_records": cleaned.total_records,
        "train_records": len(cleaned.train_records), "val_records": len(cleaned.val_records),
        "dropped_count": cleaned.dropped_count, "warnings": cleaned.warnings,
        "recipe": plan.recipe.model_dump(mode="json"), "progress": last["progress"],
        "presets": presets_payload(plan),
        "preview": [{
            "prompt": r.prompt or (r.messages[0].content if r.messages else ""),
            "completion": r.completions[0] if r.completions else (r.messages[-1].content if len(r.messages) > 1 else ""),
        } for r in cleaned.train_records[:3]],
        "environment_quality": [m.model_dump(mode="json") for m in cleaned.environment_quality],
    }


@router.post(
    "/api/dataset/upload", response_model=UploadedDataset,
    responses={202: {"model": IngestionAccepted, "description": "Accepted when `background=true`; poll the status route."}},
)
async def upload_dataset(file: Annotated[UploadFile, File()], state: State, background: bool = False) -> Any:
    name = _upload_name(file.filename)
    if any(j.get("dataset_name") == name and j["status"] == "running" for j in state.store.list("ingestion")):
        raise HTTPException(status_code=409, detail="This dataset is being processed. Wait for it to finish.")
    uploads = uploads_dir()
    uploads.mkdir(parents=True, exist_ok=True)
    dest_path = uploads / name
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds 100 MiB")
    dest_path.write_bytes(data)
    planner = state.planner()
    # The waiting request answers with the result or re-raises the failure, so its error handlers apply.
    outcome: dict[str, Any] = {}

    def work(handle: JobHandle) -> dict[str, Any]:
        try:
            outcome["result"] = inspect_upload(dest_path, planner, lambda update: handle.update(progress=update))
        except Exception as exc:
            outcome["failure"] = exc
            raise
        return {"result": outcome["result"]}

    job_id = state.ingestion.start({"dataset_name": name, "progress": {}, "error": None}, work)
    if background:
        return JSONResponse(status_code=202, content={"job_id": job_id})
    await asyncio.to_thread(state.ingestion.join, job_id)
    if "failure" in outcome:
        raise outcome["failure"]
    return outcome["result"]


@router.get("/api/dataset/status/{job_id}", response_model=IngestionJob)
async def dataset_status(job_id: str, state: State) -> dict[str, Any]:
    record = state.ingestion.snapshot(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset job not found")
    return record
