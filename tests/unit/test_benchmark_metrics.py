"""Tests for benchmark metrics: pass@1, pass@k, pass^k, Cohen's kappa and robustness."""
from __future__ import annotations

import math
import pytest

from autotune.core.domain.models import SampleResult
from autotune.infrastructure.evaluators.metrics import (
    compute_cohens_kappa,
    compute_pass_at_k,
    compute_pass_pow_k,
    _reward_hacking_risk,
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


def test_per_k_values_match_the_estimators_averaged_over_tasks():
    samples = _task("a", [True, True, False, False]) + _task("b", [True, True, True, False, False, True])

    score = robustness(samples)

    assert score.k_labels == [1, 2, 3, 4]
    for k, at_k, pow_k in zip(score.k_labels, score.pass_at_k, score.pass_pow_k, strict=True):
        assert at_k == pytest.approx((compute_pass_at_k(4, 2, k) + compute_pass_at_k(6, 4, k)) / 2, abs=1e-4)
        assert pow_k == pytest.approx((compute_pass_pow_k(4, 2, k) + compute_pass_pow_k(6, 4, k)) / 2, abs=1e-4)


def _graded(passes: int, fails: int, judge_passes: int) -> list[SampleResult]:
    """Grader passes first, then fails. The judge passes on the first `judge_passes` samples."""
    outcomes = [True] * passes + [False] * fails
    judge = [1.0 if i < judge_passes else 0.0 for i in range(len(outcomes))]
    return _task("a", outcomes, judge)


@pytest.mark.parametrize(
    ("gap", "kappa", "tier"),
    [
        (0.35, None, "low"),
        (0.3501, None, "moderate"),
        (0.6, None, "moderate"),
        (0.6001, None, "high"),
        (0.0, 0.3, "moderate"),
        (0.0, 0.2999, "high"),
        (0.0, 0.6, "low"),
        (0.0, 0.5999, "moderate"),
        (0.4, 0.9, "moderate"),
        (0.0, 1.0, "low"),
    ],
)
def test_reward_hacking_tiers_at_every_threshold(gap, kappa, tier):
    assert _reward_hacking_risk(gap, kappa) == tier


def test_a_constant_grader_gives_no_kappa_and_the_gap_sets_the_tier():
    # The grader passes everything and the judge agrees on 19 of 20 samples.
    score = robustness(_graded(passes=20, fails=0, judge_passes=19))

    assert score.grader_judge_kappa is None
    assert score.reward_hacking_risk == "low"


def test_a_constant_judge_gives_no_kappa():
    assert robustness(_graded(passes=10, fails=10, judge_passes=0)).grader_judge_kappa is None


def test_kappa_is_reported_when_both_raters_vary():
    assert robustness(_graded(passes=10, fails=10, judge_passes=10)).grader_judge_kappa == pytest.approx(1.0)
