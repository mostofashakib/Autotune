"""Write environment quality metrics as JSON and CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from autotune.core.domain.models import EnvQualityMetrics

_FIELDS = list(EnvQualityMetrics.model_fields)


def write_env_quality(metrics: list[EnvQualityMetrics], out_dir: Path) -> list[Path]:
    """Write `env_quality.json` and `env_quality.csv` into out_dir and return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [m.model_dump() for m in metrics]
    json_path = out_dir / "env_quality.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = out_dir / "env_quality.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})
    return [json_path, csv_path]
