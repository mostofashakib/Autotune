"""Every test writes jobs and uploads under its own temporary data directory."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "autotune-data"
    monkeypatch.setenv("AUTOTUNE_DATA_DIR", str(data_dir))
    return data_dir
