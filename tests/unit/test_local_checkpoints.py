"""LocalCheckpointStorage saves manifests and exports GGUF through llama.cpp."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotune.core.domain.errors import MissingExtraError
from autotune.infrastructure.storage.checkpoint import PolicyCheckpoint
from autotune.infrastructure.storage.job_store import JobStore
from autotune.infrastructure.storage.local_checkpoints import LocalCheckpointStorage
from autotune.presentation.api import app as api

UNSET_MESSAGE = "Set LLAMA_CPP_DIR to a llama.cpp checkout to export GGUF files."


@pytest.fixture
def llama_cpp(tmp_path, monkeypatch):
    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_text("")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(root))
    return root


def _stub_merge(adapter_dir, merged_dir):
    (merged_dir / "merged.marker").write_text(adapter_dir.name)


def test_save_copies_the_checkpoint_and_returns_the_manifest(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    PolicyCheckpoint(objective="sft", base_model="m", model_path=str(source), num_examples=1).save(source)
    (source / "adapter.bin").write_text("weights")

    manifest = LocalCheckpointStorage().save("job", source, target)

    assert manifest == target / "policy_checkpoint.json"
    assert (target / "adapter.bin").read_text() == "weights"


def test_save_in_place_returns_the_manifest(tmp_path):
    PolicyCheckpoint(objective="sft", base_model="m", model_path=str(tmp_path), num_examples=1).save(tmp_path)

    assert LocalCheckpointStorage().save("job", tmp_path, tmp_path) == tmp_path / "policy_checkpoint.json"


def test_export_gguf_names_llama_cpp_dir_when_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_DIR", "")
    with pytest.raises(ValueError, match=UNSET_MESSAGE):
        LocalCheckpointStorage(merge=_stub_merge).export_gguf(tmp_path, tmp_path / "out.gguf")


def test_export_gguf_rejects_a_checkout_without_the_converter(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    with pytest.raises(ValueError, match=UNSET_MESSAGE):
        LocalCheckpointStorage(merge=_stub_merge).export_gguf(tmp_path, tmp_path / "out.gguf")


def test_export_gguf_runs_converter(tmp_path, llama_cpp):
    calls = []
    merged_markers = []
    output = tmp_path / "out.gguf"

    def convert(args):
        calls.append(args)
        merged_markers.append((Path(args[1]) / "merged.marker").read_text())

    result = LocalCheckpointStorage(convert=convert, merge=_stub_merge).export_gguf(tmp_path / "ckpt", output)

    assert result == output
    assert calls[0][0] == str(llama_cpp / "convert_hf_to_gguf.py")
    assert calls[0][2:] == ["--outfile", str(output)]
    assert merged_markers == ["ckpt"]


def test_merge_without_peft_names_the_training_extra(tmp_path, llama_cpp, monkeypatch):
    monkeypatch.setitem(sys.modules, "peft", None)
    with pytest.raises(MissingExtraError, match="uv sync --extra training"):
        LocalCheckpointStorage(convert=lambda args: None).export_gguf(tmp_path, tmp_path / "out.gguf")


class FakeStorage:
    def __init__(self):
        self.exported = []

    def export_gguf(self, checkpoint_dir, output_file):
        self.exported.append((checkpoint_dir, output_file))
        return output_file


@pytest.fixture
def client(isolated_data_dir):
    return TestClient(api.create_app())


def _seed(isolated_data_dir, job_id, **fields):
    JobStore(isolated_data_dir / "autotune.db").save("training", job_id, {"job_id": job_id, "status": "completed", **fields})


def _manifest(checkpoint_dir, run_id):
    PolicyCheckpoint(objective="sft", base_model="m", model_path=str(checkpoint_dir), num_examples=1,
                     run_id=run_id).save(checkpoint_dir)


def test_export_route_starts_a_job(client, isolated_data_dir, tmp_path):
    _seed(isolated_data_dir, "job_a", checkpoint_dir=str(tmp_path / "ckpt"))
    _manifest(tmp_path / "ckpt", "job_a")
    storage = FakeStorage()
    client.app.state.autotune.checkpoints = storage

    response = client.post("/api/training/job_a/export-gguf")

    assert response.status_code == 202
    export_id = response.json()["job_id"]
    client.app.state.autotune.exports.join(export_id, 5)
    body = client.get(f"/api/exports/{export_id}").json()
    assert body["status"] == "completed"
    assert body["gguf_path"] == str(tmp_path / "ckpt" / "model.gguf")
    assert storage.exported == [(tmp_path / "ckpt", tmp_path / "ckpt" / "model.gguf")]


def test_export_route_404s_for_unknown_job(client):
    assert client.post("/api/training/job_missing/export-gguf").status_code == 404


def test_export_route_409s_without_a_checkpoint(client, isolated_data_dir):
    _seed(isolated_data_dir, "job_b", checkpoint_dir=None)
    assert client.post("/api/training/job_b/export-gguf").status_code == 409


def test_export_route_refuses_a_checkpoint_a_later_job_overwrote(client, isolated_data_dir, tmp_path):
    shared = tmp_path / "ckpt"
    _seed(isolated_data_dir, "job_a", checkpoint_dir=str(shared))
    _seed(isolated_data_dir, "job_b", checkpoint_dir=str(shared))
    _manifest(shared, "job_b")
    client.app.state.autotune.checkpoints = FakeStorage()

    stale = client.post("/api/training/job_a/export-gguf")
    current = client.post("/api/training/job_b/export-gguf")

    assert stale.status_code == 409
    assert stale.json()["detail"] == (f"Checkpoint in {shared} now belongs to job job_b. "
                                      "Re-run training for job_a or export the newer job.")
    assert current.status_code == 202
    client.app.state.autotune.exports.join(current.json()["job_id"], 5)


def test_export_route_409s_when_the_checkpoint_has_no_manifest(client, isolated_data_dir, tmp_path):
    _seed(isolated_data_dir, "job_a", checkpoint_dir=str(tmp_path / "ckpt"))
    client.app.state.autotune.checkpoints = FakeStorage()

    response = client.post("/api/training/job_a/export-gguf")

    assert response.status_code == 409
    assert "Re-run training for job_a" in response.json()["detail"]


def test_export_status_404s_for_unknown_export(client):
    assert client.get("/api/exports/export-missing").status_code == 404


def test_an_export_cut_off_by_a_restart_shows_as_interrupted(isolated_data_dir):
    JobStore(isolated_data_dir / "autotune.db").save("export", "export-old", {
        "job_id": "export-old", "status": "running", "created_at": "2026-11-06T00:00:00+00:00",
        "checkpoint_dir": "ckpt", "gguf_path": None, "error": None, "logs": []})

    body = TestClient(api.create_app()).get("/api/exports/export-old").json()

    assert body["status"] == "interrupted"
