"""Export a trained checkpoint to GGUF and watch the export."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from autotune.infrastructure.storage.checkpoint import PolicyCheckpoint
from autotune.presentation.api.jobs import JobHandle
from autotune.presentation.api.schemas import GgufExport, JobStarted
from autotune.presentation.api.state import State

router = APIRouter()

GGUF_NAME = "model.gguf"


@router.post("/api/training/{job_id}/export-gguf", status_code=202, response_model=JobStarted)
async def export_gguf(job_id: str, state: State) -> dict[str, Any]:
    record = state.training.snapshot(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not record.get("checkpoint_dir"):
        raise HTTPException(status_code=409, detail=f"Job {job_id} has no checkpoint to export")
    checkpoint_dir = Path(record["checkpoint_dir"])
    _require_owned_by(checkpoint_dir, job_id)
    output_file = checkpoint_dir / GGUF_NAME

    def work(handle: JobHandle) -> dict[str, Any]:
        handle.log(f"Exporting {checkpoint_dir} to GGUF")
        written = state.checkpoints.export_gguf(checkpoint_dir, output_file)
        handle.log(f"GGUF written to {written}")
        return {"gguf_path": str(written)}

    export_id = state.exports.start(
        {"training_job_id": job_id, "checkpoint_dir": str(checkpoint_dir), "gguf_path": None, "error": None}, work)
    return {"job_id": export_id, "status": "running"}


def _require_owned_by(checkpoint_dir: Path, job_id: str) -> None:
    """Refuse to export weights a later job wrote over this job's shared checkpoint dir."""
    try:
        owner = PolicyCheckpoint.load(checkpoint_dir).run_id
    except FileNotFoundError:
        raise HTTPException(status_code=409, detail=(
            f"Checkpoint in {checkpoint_dir} has no manifest naming its job. Re-run training for {job_id}.")) from None
    if owner != job_id:
        raise HTTPException(status_code=409, detail=(
            f"Checkpoint in {checkpoint_dir} now belongs to job {owner}. "
            f"Re-run training for {job_id} or export the newer job."))


@router.get("/api/exports/{export_id}", response_model=GgufExport)
async def get_export(export_id: str, state: State) -> dict[str, Any]:
    record = state.exports.snapshot(export_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Export not found")
    return record
