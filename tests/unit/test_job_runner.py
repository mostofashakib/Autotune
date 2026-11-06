import logging
import threading

import pytest

from autotune.infrastructure.storage.job_store import JobStore
from autotune.presentation.api.jobs import JobHandle, JobRunner


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs.db")


@pytest.fixture
def runner(store):
    return JobRunner(store, "training", id_prefix="train")


def test_successful_work_completes(runner, store):
    def work(handle: JobHandle):
        handle.update(progress=5)
        handle.log("hello")
        return {"result": "ok"}

    job_id = runner.start({"model": "m"}, work)
    runner.join(job_id)

    record = store.get("training", job_id)
    assert job_id.startswith("train")
    assert record["status"] == "completed"
    assert record["model"] == "m"
    assert record["progress"] == 5
    assert record["result"] == "ok"
    assert record["logs"] == ["hello"]
    assert record["created_at"]
    assert not runner.live(job_id)
    assert runner.snapshot(job_id) == record


def test_expected_failure_is_recorded(runner, store, caplog):
    def work(handle):
        raise ValueError("bad input")

    with caplog.at_level(logging.WARNING):
        job_id = runner.start({}, work)
        runner.join(job_id)

    record = store.get("training", job_id)
    assert record["status"] == "failed"
    assert record["error"] == "bad input"
    assert [r.levelno for r in caplog.records if "bad input" in r.getMessage()] == [logging.WARNING]


def test_unexpected_failure_is_logged_and_recorded(runner, store, caplog, monkeypatch):
    escaped = []
    monkeypatch.setattr(threading, "excepthook", lambda args: escaped.append(args.exc_value))

    def work(handle):
        raise KeyError("missing")

    with caplog.at_level(logging.WARNING):
        job_id = runner.start({}, work)
        runner.join(job_id)

    record = store.get("training", job_id)
    assert record["status"] == "failed"
    assert record["error"] == "'missing'"
    assert any(r.exc_info for r in caplog.records)
    assert escaped == []


def test_work_can_set_cancelled(runner, store):
    job_id = runner.start({}, lambda handle: {"status": "cancelled"})
    runner.join(job_id)

    assert store.get("training", job_id)["status"] == "cancelled"


def test_snapshot_is_isolated_from_later_updates(runner):
    logged, release = threading.Event(), threading.Event()

    def work(handle):
        handle.log("one")
        logged.set()
        release.wait(5)
        handle.log("two")

    job_id = runner.start({}, work)
    logged.wait(5)
    snap = runner.snapshot(job_id)
    assert runner.live(job_id)
    release.set()
    runner.join(job_id)

    assert snap["logs"] == ["one"]
    assert snap["status"] == "running"
    assert runner.snapshot(job_id)["logs"] == ["one", "two"]


def test_concurrent_logs_are_not_lost(runner, store):
    def work(handle):
        def spam(tag):
            for i in range(500):
                handle.log(f"{tag}{i}")

        threads = [threading.Thread(target=spam, args=(tag,)) for tag in "ab"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    job_id = runner.start({}, work)
    runner.join(job_id)

    assert len(store.get("training", job_id)["logs"]) == 1000


def test_unknown_job_has_no_snapshot(runner):
    assert runner.snapshot("nope") is None
    assert not runner.live("nope")
