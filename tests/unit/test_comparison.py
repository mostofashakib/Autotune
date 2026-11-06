"""Comparing two models reports only what their samples measured."""
from __future__ import annotations

import pytest

from autotune.core.domain.models import ComparisonReport, ModelEndpoint, SampleResult
from autotune.infrastructure.evaluators.comparison import compare

BASE = ModelEndpoint(base_url="http://localhost:8000/v1", model="base")
TRAINED = ModelEndpoint(base_url="http://localhost:8001/v1", model="trained", api_key_env="TRAINED_KEY")


def _samples(task_id: str, outcomes: list[bool], score: float = 0.5) -> list[SampleResult]:
    return [SampleResult(task_id=task_id, passed=passed, score=score) for passed in outcomes]


def test_pass_at_k_is_the_unbiased_estimate_averaged_over_tasks():
    base = _samples("a", [True, False, False]) + _samples("b", [False, False, False])
    trained = _samples("a", [True, True, False]) + _samples("b", [True, False, False])

    report = compare("dataset", 3, BASE, TRAINED, base, trained)

    assert report.base.pass_at_1 == pytest.approx((1 / 3 + 0) / 2)
    assert report.trained.pass_at_1 == pytest.approx((2 / 3 + 1 / 3) / 2)
    assert report.base.pass_at_k == pytest.approx(0.5)
    assert report.trained.pass_at_k == pytest.approx(1.0)
    assert report.delta_pass_at_1 == pytest.approx(report.trained.pass_at_1 - report.base.pass_at_1)
    assert report.delta_pass_at_k == pytest.approx(0.5)


def test_each_task_reports_both_models_and_their_difference():
    base = _samples("a", [False, False], score=0.2)
    trained = _samples("a", [True, False], score=0.6)

    task = compare("harbor", 2, BASE, TRAINED, base, trained).tasks[0]

    assert task.task_id == "a"
    assert (task.base.samples, task.base.passed) == (2, 0)
    assert (task.trained.samples, task.trained.passed) == (2, 1)
    assert task.delta_pass_at_1 == pytest.approx(0.5)
    assert task.delta_mean_score == pytest.approx(0.4)


def test_errored_samples_count_as_failures_and_are_reported():
    base = [SampleResult(task_id="a", passed=False, score=0.0, error="trial crashed")] + _samples("a", [True])
    trained = _samples("a", [True, True])

    report = compare("harbor", 2, BASE, TRAINED, base, trained)

    assert report.base.errors == 1
    assert report.base.passed == 1
    assert report.trained.errors == 0


def test_judge_scores_are_averaged_only_when_present():
    base = [SampleResult(task_id="a", passed=True, score=1.0, judge_score=0.4)]
    trained = [SampleResult(task_id="a", passed=True, score=1.0, judge_score=0.8)]
    without_judge = compare("dataset", 1, BASE, TRAINED, _samples("a", [True]), _samples("a", [True]))

    report = compare("dataset", 1, BASE, TRAINED, base, trained)

    assert report.trained.mean_judge_score == pytest.approx(0.8)
    assert without_judge.base.mean_judge_score is None


def test_models_measured_on_different_tasks_cannot_be_compared():
    with pytest.raises(ValueError, match="same tasks"):
        compare("dataset", 1, BASE, TRAINED, _samples("a", [True]), _samples("b", [True]))


def test_a_model_without_samples_cannot_be_compared():
    with pytest.raises(ValueError, match="no samples"):
        compare("dataset", 1, BASE, TRAINED, [], [])


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k"):
        compare("dataset", 0, BASE, TRAINED, _samples("a", [True]), _samples("a", [True]))


def test_the_report_never_carries_an_api_key(monkeypatch):
    monkeypatch.setenv("TRAINED_KEY", "sk-secret-value")

    dumped = compare("dataset", 1, BASE, TRAINED, _samples("a", [True]), _samples("a", [True])).model_dump_json()

    assert "sk-secret-value" not in dumped
    assert "TRAINED_KEY" in dumped


class RecordingRunner:
    def __init__(self, by_model):
        self.by_model = by_model
        self.models = []

    def run(self, model):
        self.models.append(model.model)
        return self.by_model[model.model]


def test_run_comparison_measures_base_then_trained_on_the_same_runner():
    from autotune.infrastructure.evaluators.comparison import run_comparison

    runner = RecordingRunner({"base": _samples("a", [False]), "trained": _samples("a", [True])})

    report = run_comparison("dataset", runner, 1, BASE, TRAINED)

    assert runner.models == ["base", "trained"]
    assert report.delta_pass_at_1 == pytest.approx(1.0)


def test_write_report_saves_the_measured_report_as_json(tmp_path):
    from autotune.infrastructure.evaluators.comparison import write_report

    report = compare("dataset", 1, BASE, TRAINED, _samples("a", [False]), _samples("a", [True]))

    path = write_report(report, tmp_path / "out")

    assert path == tmp_path / "out" / "benchmark_report.json"
    assert path.read_text() == report.model_dump_json(indent=2)


def test_comparison_report_carries_both_robustness_scores():
    base = _samples("a", [True, False, False]) + _samples("b", [True, True, True])
    trained = _samples("a", [True, True, True]) + _samples("b", [True, True, False])

    report = compare("dataset", 3, BASE, TRAINED, base, trained)

    assert report.base_robustness.k_labels == [1, 2, 3]
    assert report.base_robustness.pass_at_k[0] == pytest.approx(report.base.pass_at_1, abs=1e-3)
    assert report.trained_robustness.pass_pow_k[-1] == pytest.approx(0.5)
    assert report.base_robustness.grader_judge_kappa is None


def test_a_report_stored_before_robustness_still_validates():
    report = compare("dataset", 1, BASE, TRAINED, _samples("a", [True]), _samples("a", [True]))
    legacy = report.model_dump(mode="json", exclude={"base_robustness", "trained_robustness"})

    loaded = ComparisonReport.model_validate(legacy)

    assert loaded.base_robustness is None
    assert loaded.trained_robustness is None
