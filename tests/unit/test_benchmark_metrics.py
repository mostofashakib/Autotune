"""Tests for benchmark metrics: pass@1, pass@k, pass^k, Cohen's kappa and robustness."""
from __future__ import annotations

import math
import pytest

from autotune.core.domain.models import SampleResult
from autotune.infrastructure.evaluators.metrics import (
    compute_cohens_kappa,
    compute_pass_at_k,
    compute_pass_pow_k,
    robustness,
)


def test_pass_at_k_and_pass_pow_k_computations():
    assert compute_pass_at_k(10, 0, 1) == 0.0
    assert compute_pass_pow_k(10, 0, 1) == 0.0

    assert compute_pass_at_k(10, 10, 1) == 1.0
    assert compute_pass_at_k(10, 10, 5) == 1.0
    assert compute_pass_pow_k(10, 10, 1) == 1.0
    assert compute_pass_pow_k(10, 10, 5) == 1.0

    assert math.isclose(compute_pass_at_k(10, 4, 1), 0.4)
    assert math.isclose(compute_pass_pow_k(10, 4, 1), 0.4)

    pass_at_3 = compute_pass_at_k(10, 4, 3)
    pass_pow_3 = compute_pass_pow_k(10, 4, 3)
    assert pass_at_3 > 0.4
    assert pass_pow_3 < 0.4

    assert compute_pass_pow_k(10, 2, 3) == 0.0


def test_cohens_kappa_agreement():
    assert compute_cohens_kappa([], []) == 0.0
    assert compute_cohens_kappa([True], [True, False]) == 0.0

    eval1 = [True, False, True, True, False, False, True, False]
    eval2 = [True, False, True, True, False, False, True, False]
    assert math.isclose(compute_cohens_kappa(eval1, eval2), 1.0)

    eval_a = [True, True, True, True, False, False, False, False]
    eval_b = [True, False, True, False, True, False, True, False]
    assert compute_cohens_kappa(eval_a, eval_b) < 0.2


def _task(task_id: str, outcomes: list[bool], judge: list[float] | None = None) -> list[SampleResult]:
    judge = judge if judge is not None else [None] * len(outcomes)
    return [
        SampleResult(task_id=task_id, passed=passed, score=float(passed), judge_score=j)
        for passed, j in zip(outcomes, judge, strict=True)
    ]


def test_robustness_curves_span_available_samples():
    samples = [s for t in "abc" for s in _task(t, [True, False, True, False])]

    score = robustness(samples)

    assert score.k_labels == [1, 2, 3, 4]
    assert score.pass_at_k[0] == pytest.approx(0.5)
    assert score.pass_pow_k[0] == pytest.approx(0.5)
    assert score.pass_pow_k[-1] == 0.0
    assert score.pass_at_k[-1] == 1.0


def test_robustness_uses_the_smallest_sample_count_up_to_ten():
    samples = _task("a", [True] * 12) + _task("b", [True] * 3)

    assert robustness(samples).k_labels == [1, 2, 3]
    assert robustness(_task("a", [True] * 12)).k_labels == list(range(1, 11))


def test_single_sample_tasks_give_one_point():
    score = robustness(_task("a", [True]) + _task("b", [False]))

    assert score.k_labels == [1]
    assert score.pass_at_k == [0.5]
    assert score.pass_pow_k == [0.5]


def test_kappa_is_none_without_a_judge():
    assert robustness(_task("a", [True, False])).grader_judge_kappa is None


def test_kappa_is_none_when_any_sample_lacks_a_judge_score():
    samples = _task("a", [True, False], [1.0, 0.0]) + _task("b", [True, False])

    assert robustness(samples).grader_judge_kappa is None


def test_kappa_compares_grader_with_judge():
    samples = _task("a", [True, False, True, False], [0.9, 0.1, 0.5, 0.2])

    assert robustness(samples).grader_judge_kappa == pytest.approx(1.0)


def test_disagreeing_judge_raises_reward_hacking_risk():
    samples = _task("a", [True, True, False, False], [0.0, 0.0, 1.0, 1.0])

    score = robustness(samples)

    assert score.grader_judge_kappa == pytest.approx(-1.0)
    assert score.reward_hacking_risk == "high"


def test_large_pass_gap_raises_reward_hacking_risk():
    samples = [s for t in "abcd" for s in _task(t, [True] + [False] * 9)]

    assert robustness(samples).reward_hacking_risk == "high"


def test_consistent_results_have_low_reward_hacking_risk():
    assert robustness(_task("a", [True] * 4)).reward_hacking_risk == "low"


def test_unmeasured_risks_are_none_not_low():
    score = robustness(_task("a", [True, False]))

    assert score.memorization_risk is None
    assert score.contamination_risk is None


@pytest.mark.parametrize(("n", "c", "k"), [(5, -1, 1), (5, 6, 1), (5, 2, 0), (-1, 0, 1)])
def test_rejects_impossible_counts(n, c, k):
    with pytest.raises(ValueError):
        compute_pass_at_k(n=n, c=c, k=k)


def test_rejects_invalid_sample_count():
    with pytest.raises((ValueError, ZeroDivisionError)):
        compute_pass_at_k(n=0, c=0, k=1)


def test_robustness_refuses_to_rate_risk_without_any_samples():
    with pytest.raises(ValueError, match="no samples"):
        robustness([])
