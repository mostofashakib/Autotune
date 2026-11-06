import math
from pathlib import Path

import pytest

LIMIT_LITERALS = ("2e-5", "100 * 1024 * 1024", "20 * 1024 * 1024")


@pytest.mark.parametrize("vram", [-1, math.nan, math.inf])
def test_require_valid_vram_rejects_invalid_values(vram):
    from autotune.core.domain.limits import require_valid_vram

    with pytest.raises(ValueError, match="VRAM must be a finite nonnegative number"):
        require_valid_vram(vram)


@pytest.mark.parametrize("vram", [0.0, 24.0])
def test_require_valid_vram_returns_valid_values_unchanged(vram):
    from autotune.core.domain.limits import require_valid_vram

    assert require_valid_vram(vram) == vram


def test_no_magic_limits_outside_limits_module():
    source_root = Path(__file__).parents[2] / "src" / "autotune"
    offenders = [
        f"{path.relative_to(source_root)}: {literal}"
        for path in source_root.rglob("*.py")
        if path.name != "limits.py"
        for literal in LIMIT_LITERALS
        if literal in path.read_text()
    ]
    assert offenders == []
