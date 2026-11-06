"""Environment quality report files."""
import csv
import json

from autotune.core.domain.models import EnvQualityMetrics
from autotune.infrastructure.evaluators.report import write_env_quality


def _metrics() -> list[EnvQualityMetrics]:
    return [
        EnvQualityMetrics(env_name="email_env", state_coverage_score=0.9, reward_density=0.7, dead_end_rate=0.1, action_diversity=0.6, num_episodes=10, num_steps=100),
        EnvQualityMetrics(env_name="pm_env", state_coverage_score=None, reward_density=0.6, dead_end_rate=0.2, action_diversity=0.5, num_episodes=10, num_steps=90),
    ]


def test_write_env_quality_writes_json_and_csv(tmp_path):
    paths = write_env_quality(_metrics(), tmp_path)
    assert paths == [tmp_path / "env_quality.json", tmp_path / "env_quality.csv"]
    data = json.loads(paths[0].read_text())
    assert [row["env_name"] for row in data] == ["email_env", "pm_env"]
    assert data[1]["state_coverage_score"] is None
    with paths[1].open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["reward_density"] == "0.7"
    assert rows[1]["state_coverage_score"] == ""


def test_write_env_quality_creates_the_directory(tmp_path):
    write_env_quality(_metrics(), tmp_path / "nested" / "out")
    assert (tmp_path / "nested" / "out" / "env_quality.csv").exists()


def test_write_env_quality_with_no_metrics_writes_empty_files(tmp_path):
    paths = write_env_quality([], tmp_path)
    assert json.loads(paths[0].read_text()) == []
    assert paths[1].read_text().startswith("env_name,")
