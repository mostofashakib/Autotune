"""Benchmark evaluation metrics (Feature Request 9).

Includes:
- pass@1, pass@k (unbiased combinatorial estimator)
- pass^k (all-k pass / robustness estimator)
- Cohen's kappa agreement rate
- robustness(): the pass curves, grader/judge agreement, and reward-hacking risk
  for one model's samples.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, Field

from autotune.core.domain.models import Risk, RobustnessScore, SampleResult

MAX_K = 10
JUDGE_PASS_THRESHOLD = 0.5


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute the unbiased estimator of pass@k (HumanEval / Chen et al.).

    Args:
        n: Total number of samples generated per problem.
        c: Number of correct (passing) samples.
        k: Sample threshold (k <= n).

    Raises:
        ValueError: when n or k is not positive, or c is outside [0, n].
    """
    if n <= 0 or k <= 0:
        raise ValueError(f"pass@k needs n >= 1 and k >= 1, got n={n}, k={k}")
    if not 0 <= c <= n:
        raise ValueError(f"pass@k needs 0 <= c <= n, got c={c}, n={n}")
    if n < k:
        return float(c >= 1)
    if c >= n:
        return 1.0
    if n - c < k:
        return 1.0
    comb_total = math.comb(n, k)
    if comb_total == 0:
        return 0.0
    comb_fail = math.comb(n - c, k)
    return 1.0 - (comb_fail / comb_total)


def compute_pass_pow_k(n: int, c: int, k: int) -> float:
    """Compute pass^k (probability that all k drawn samples pass).

    Measures consistency and resilience against reward hacking / lucky guessing.
    """
    if n <= 0 or k <= 0:
        return 0.0
    if n < k or c < k:
        return 0.0
    comb_total = math.comb(n, k)
    if comb_total == 0:
        return 0.0
    comb_pass = math.comb(c, k)
    return comb_pass / comb_total


def compute_cohens_kappa(
    eval1: list[bool | int], eval2: list[bool | int]
) -> float:
    """Compute Cohen's kappa agreement rate between two evaluators/runs on the same task items.

    kappa = (p_o - p_e) / (1 - p_e)
    """
    if not eval1 or len(eval1) != len(eval2):
        return 0.0

    total = len(eval1)
    b1 = [bool(x) for x in eval1]
    b2 = [bool(x) for x in eval2]

    # Contingency counts
    yy = sum(1 for x, y in zip(b1, b2, strict=True) if x and y)
    yn = sum(1 for x, y in zip(b1, b2, strict=True) if x and not y)
    ny = sum(1 for x, y in zip(b1, b2, strict=True) if not x and y)
    nn = sum(1 for x, y in zip(b1, b2, strict=True) if not x and not y)

    p_o = (yy + nn) / total

    # Marginal probabilities
    p1_yes = (yy + yn) / total
    p1_no = 1.0 - p1_yes
    p2_yes = (yy + ny) / total
    p2_no = 1.0 - p2_yes

    p_e = (p1_yes * p2_yes) + (p1_no * p2_no)

    if math.isclose(1.0, p_e):
        return 1.0 if math.isclose(p_o, 1.0) else 0.0

    return (p_o - p_e) / (1.0 - p_e)


class TaskTrial(BaseModel):
    task_id: str
    passed_samples: int = Field(..., ge=0)
    total_samples: int = Field(..., ge=1)
    # The reference grader's verdicts, and the judge's, one per sample.
    # The judge list is empty unless every sample was judged.
    verifier_passes: list[bool] = Field(default_factory=list)
    ground_truth_passes: list[bool] = Field(default_factory=list)


def robustness(samples: Sequence[SampleResult]) -> RobustnessScore:
    """Pass curves, grader/judge agreement and reward-hacking risk for one model.

    Memorization and contamination need paraphrased items and n-gram overlap,
    which Autotune never collects, so those risks stay None.
    """
    if not samples:
        # No samples means no measurement, not a low risk.
        raise ValueError("no samples to measure robustness from")
    trials = _trials(samples)
    k_labels = list(range(1, min(min(t.total_samples for t in trials), MAX_K) + 1))
    pass_at_k, pass_pow_k = _pass_curves(trials, k_labels)
    kappa = _kappa(trials) if all(s.judge_score is not None for s in samples) else None
    return RobustnessScore(
        k_labels=k_labels,
        pass_at_k=pass_at_k,
        pass_pow_k=pass_pow_k,
        grader_judge_kappa=kappa,
        reward_hacking_risk=_reward_hacking_risk(pass_at_k[-1] - pass_pow_k[-1], kappa),
        memorization_risk=None,
        contamination_risk=None,
    )


def _trials(samples: Sequence[SampleResult]) -> list[TaskTrial]:
    by_task: dict[str, list[SampleResult]] = defaultdict(list)
    for sample in samples:
        by_task[sample.task_id].append(sample)
    judged = all(s.judge_score is not None for s in samples)
    return [
        TaskTrial(
            task_id=task_id,
            passed_samples=sum(s.passed for s in task_samples),
            total_samples=len(task_samples),
            verifier_passes=[s.passed for s in task_samples],
            ground_truth_passes=[s.judge_score >= JUDGE_PASS_THRESHOLD for s in task_samples] if judged else [],
        )
        for task_id, task_samples in by_task.items()
    ]


def _pass_curves(trials: list[TaskTrial], k_labels: list[int]) -> tuple[list[float], list[float]]:
    """Mean pass@k and pass^k across tasks, for each k."""
    pass_at_k = [
        round(sum(compute_pass_at_k(t.total_samples, t.passed_samples, k) for t in trials) / len(trials), 4)
        for k in k_labels
    ]
    pass_pow_k = [
        round(sum(compute_pass_pow_k(t.total_samples, t.passed_samples, k) for t in trials) / len(trials), 4)
        for k in k_labels
    ]
    return pass_at_k, pass_pow_k


def _kappa(trials: list[TaskTrial]) -> float | None:
    """Kappa between grader and judge, or None when either rater never varies.

    A rater that gives one answer throughout pins kappa at 0 however closely the
    other one agrees, so the number would mislead.
    """
    verifier = [passed for t in trials for passed in t.verifier_passes]
    ground_truth = [passed for t in trials for passed in t.ground_truth_passes]
    if len(set(verifier)) < 2 or len(set(ground_truth)) < 2:
        return None
    return round(compute_cohens_kappa(verifier, ground_truth), 4)


def _reward_hacking_risk(gap: float, kappa: float | None) -> Risk:
    """A large gap between pass@k and pass^k, or a judge that disagrees with the grader."""
    if gap > 0.6 or (kappa is not None and kappa < 0.3):
        return "high"
    if gap > 0.35 or (kappa is not None and kappa < 0.6):
        return "moderate"
    return "low"
