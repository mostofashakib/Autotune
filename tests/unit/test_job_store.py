"""Training, benchmark, and pod records persist in SQLite across restarts."""
from __future__ import annotations

import threading

from autotune import settings
from autotune.infrastructure.storage.job_store import JobStore


def test_the_data_dir_comes_from_the_environment(isolated_data_dir):
    assert settings.data_dir() == isolated_data_dir


def test_the_data_dir_defaults_inside_the_repo(monkeypatch):
    monkeypatch.delenv("AUTOTUNE_DATA_DIR")
    assert settings.data_dir() == settings.ROOT / "data"


def test_a_saved_record_reads_back(tmp_path):
    store = JobStore(tmp_path / "jobs.db")

    store.save("training", "job_1", {"status": "running", "logs": ["started"]})

    assert store.get("training", "job_1") == {"status": "running", "logs": ["started"]}


def test_saving_again_replaces_the_record(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    store.save("training", "job_1", {"status": "running"})

    store.save("training", "job_1", {"status": "completed"})

    assert store.get("training", "job_1") == {"status": "completed"}
    assert len(store.list("training")) == 1


def test_an_unknown_record_is_none(tmp_path):
    assert JobStore(tmp_path / "jobs.db").get("training", "missing") is None


def test_kinds_are_kept_apart(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    store.save("training", "same", {"kind": "training"})
    store.save("benchmark", "same", {"kind": "benchmark"})

    assert store.get("benchmark", "same") == {"kind": "benchmark"}
    assert [r["kind"] for r in store.list("training")] == ["training"]


def test_list_returns_newest_first(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    for job_id in ("first", "second", "third"):
        store.save("training", job_id, {"id": job_id})

    assert [r["id"] for r in store.list("training")] == ["third", "second", "first"]


def test_updating_a_record_keeps_its_place_in_the_list(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    store.save("training", "first", {"id": "first", "n": 1})
    store.save("training", "second", {"id": "second"})

    store.save("training", "first", {"id": "first", "n": 2})

    assert [r["id"] for r in store.list("training")] == ["second", "first"]


def test_records_survive_a_new_store_on_the_same_file(tmp_path):
    JobStore(tmp_path / "jobs.db").save("benchmark", "b1", {"status": "completed"})

    assert JobStore(tmp_path / "jobs.db").get("benchmark", "b1") == {"status": "completed"}


def test_the_store_creates_its_directory(tmp_path):
    JobStore(tmp_path / "nested" / "dir" / "jobs.db").save("pod", "p1", {})

    assert (tmp_path / "nested" / "dir" / "jobs.db").is_file()


def test_running_records_become_interrupted_with_a_reason(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    store.save("training", "live", {"status": "running", "logs": []})
    store.save("training", "done", {"status": "completed", "logs": []})

    store.mark_interrupted("training")

    live = store.get("training", "live")
    assert live["status"] == "interrupted"
    assert "restart" in live["error"]
    assert live["logs"][-1] == live["error"]
    assert store.get("training", "done")["status"] == "completed"


def test_concurrent_writers_do_not_lose_records(tmp_path):
    store = JobStore(tmp_path / "jobs.db")

    threads = [threading.Thread(target=store.save, args=("training", f"j{i}", {"i": i})) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(store.list("training")) == 20
