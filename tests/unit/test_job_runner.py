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


def test_expected_failure_is_recorded(runner, store, caplog, monkeypatch):
    escaped = []
    monkeypatch.setattr(threading, "excepthook", lambda args: escaped.append(args.exc_value))

    def work(handle):
        raise ValueError("bad input")

    with caplog.at_level(logging.WARNING):
        job_id = runner.start({}, work)
        runner.join(job_id)

    record = store.get("training", job_id)
    assert record["status"] == "failed"
    assert record["error"] == "bad input"
    assert [r.levelno for r in caplog.records if "bad input" in r.getMessage()] == [logging.WARNING]
    assert escaped == []
    assert not runner.live(job_id)


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


class FailingStore(JobStore):
    """A store whose save raises for records matching `should_fail`."""

    def __init__(self, path, should_fail):
        super().__init__(path)
        self._should_fail = should_fail

    def save(self, kind, record_id, record):
        if self._should_fail(record):
            raise OSError("disk full")
        super().save(kind, record_id, record)


@pytest.fixture
def escaped(monkeypatch):
    seen = []
    monkeypatch.setattr(threading, "excepthook", lambda args: seen.append(args.exc_value))
    return seen


def test_record_uses_job_id_key(runner):
    release = threading.Event()
    job_id = runner.start({}, lambda handle: release.wait(5))

    assert runner.snapshot(job_id)["job_id"] == job_id
    assert "id" not in runner.snapshot(job_id)
    release.set()
    runner.join(job_id)
    assert runner.snapshot(job_id)["job_id"] == job_id


def test_unsaveable_update_fails_the_job_cleanly(runner, store, escaped):
    job_id = runner.start({}, lambda handle: handle.update(obj=object()))
    runner.join(job_id)

    record = store.get("training", job_id)
    assert record["status"] == "failed"
    assert "obj" not in record
    assert not runner.live(job_id)
    assert escaped == []


def test_final_save_failure_does_not_escape_the_thread(tmp_path, escaped, caplog):
    store = FailingStore(tmp_path / "jobs.db", lambda record: record["status"] != "running")
    runner = JobRunner(store, "training", id_prefix="train")

    with caplog.at_level(logging.ERROR):
        job_id = runner.start({}, lambda handle: {"result": 1})
        runner.join(job_id)

    assert escaped == []
    assert not runner.live(job_id)
    assert any(r.exc_info for r in caplog.records)


def test_failed_first_save_leaves_nothing_live(tmp_path):
    store = FailingStore(tmp_path / "jobs.db", lambda record: True)
    runner = JobRunner(store, "training", id_prefix="train")
    calls = []

    with pytest.raises(OSError):
        runner.start({}, calls.append)

    assert calls == []
    assert runner._live == {}
    assert runner._threads == {}


def test_finished_threads_are_pruned_and_join_still_returns(runner):
    job_id = runner.start({}, lambda handle: None)
    runner.join(job_id)
    runner.join(job_id)

    assert runner._threads == {}
