from pathlib import Path
from autotune.infrastructure.evaluators.side_by_side import SideBySideEvaluator


def test_evaluator_report_generation(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir(parents=True)

    evaluator = SideBySideEvaluator()
    report = evaluator.evaluate(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        checkpoint_dir=checkpoint_dir,
    )

    assert report.base_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert report.pass_at_1 > 0.0
    assert report.improvement_percentage > 0.0
    assert len(report.metrics) >= 3
    assert "improvement" in report.summary
