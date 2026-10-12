"""Turn two models' graded samples into a side-by-side report."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

from autotune.core.domain.models import (
    BenchmarkMode,
    ComparisonReport,
    ModelEndpoint,
    ModelScore,
    SampleResult,
    TaskComparison,
)
from autotune.core.ports.evaluator import BenchmarkRunner
from autotune.infrastructure.evaluators.metrics import compute_pass_at_k

REPORT_NAME = "benchmark_report.json"


def run_comparison(
    mode: BenchmarkMode, runner: BenchmarkRunner, k: int, base: ModelEndpoint, trained: ModelEndpoint,
) -> ComparisonReport:
    """Measure base, then trained, on the same runner and compare them."""
    return compare(mode, k, base, trained, runner.run(base), runner.run(trained))


def write_report(report: ComparisonReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / REPORT_NAME
    path.write_text(report.model_dump_json(indent=2))
    return path


def compare(
    mode: BenchmarkMode,
    k: int,
    base_endpoint: ModelEndpoint,
    trained_endpoint: ModelEndpoint,
    base_samples: Sequence[SampleResult],
    trained_samples: Sequence[SampleResult],
) -> ComparisonReport:
    """Score both models per task and overall. Raises when they are not comparable."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    base_by_task, trained_by_task = _by_task(base_samples), _by_task(trained_samples)
    if not base_by_task or not trained_by_task:
        raise ValueError("a model with no samples cannot be compared")
    if base_by_task.keys() != trained_by_task.keys():
        raise ValueError("base and trained must be measured on the same tasks")

    tasks = []
    for task_id in sorted(base_by_task):
        base, trained = _score([base_by_task[task_id]], k), _score([trained_by_task[task_id]], k)
        tasks.append(TaskComparison(
            task_id=task_id, base=base, trained=trained,
            delta_pass_at_1=trained.pass_at_1 - base.pass_at_1,
            delta_mean_score=trained.mean_score - base.mean_score,
        ))
    base, trained = _score(list(base_by_task.values()), k), _score(list(trained_by_task.values()), k)
    return ComparisonReport(
        mode=mode, k=k, base_endpoint=base_endpoint, trained_endpoint=trained_endpoint,
        base=base, trained=trained, tasks=tasks,
        delta_pass_at_1=trained.pass_at_1 - base.pass_at_1,
        delta_pass_at_k=trained.pass_at_k - base.pass_at_k,
        delta_mean_score=trained.mean_score - base.mean_score,
    )


def _by_task(samples: Sequence[SampleResult]) -> dict[str, list[SampleResult]]:
    grouped: dict[str, list[SampleResult]] = defaultdict(list)
    for sample in samples:
        grouped[sample.task_id].append(sample)
    return dict(grouped)


def _score(task_samples: list[list[SampleResult]], k: int) -> ModelScore:
    """pass@1 and pass@k are averaged over tasks. Counts and means are over samples."""
    flat = [sample for samples in task_samples for sample in samples]
    judged = [sample.judge_score for sample in flat if sample.judge_score is not None]
    passes = [sum(sample.passed for sample in samples) for samples in task_samples]
    return ModelScore(
        samples=len(flat),
        passed=sum(passes),
        errors=sum(sample.error is not None for sample in flat),
        pass_at_1=fmean(compute_pass_at_k(len(s), c, 1) for s, c in zip(task_samples, passes)),
        pass_at_k=fmean(compute_pass_at_k(len(s), c, k) for s, c in zip(task_samples, passes)),
        mean_score=fmean(sample.score for sample in flat),
        mean_judge_score=fmean(judged) if judged else None,
    )
