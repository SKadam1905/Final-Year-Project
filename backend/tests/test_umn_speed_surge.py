"""Speed-surge evaluation helpers (scripts/umn_speed_surge.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[2] / "scripts"


@pytest.fixture
def surge_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("umn_speed_surge", SCRIPTS / "umn_speed_surge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_surge_is_speed_over_the_recent_median_with_a_floor(surge_module):
    speeds = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 0.0, 0.2, 0.2, 0.2]
    out = surge_module.with_surge([{"speed": v} for v in speeds], history=5)
    assert out[0]["surge"] == 1.0  # no history yet
    assert out[2]["surge"] == pytest.approx(2.0)  # median(1, 1) = 1
    assert out[7]["surge"] == pytest.approx(3.0)  # median of the previous 5 readings = 1
    assert out[11]["surge"] == pytest.approx(0.2 / 0.5)  # median(1, 3, 0, 0.2, 0.2) = 0.2 < floor 0.5


def test_time_matched_auc_ignores_position_in_the_clip(surge_module):
    # Panic late in clips with a score that only grows with time: plain ranking would call it perfect.
    normal = [(t, t) for t in np.arange(0, 20, 0.5)]
    panic = [(t, t) for t in np.arange(15, 20, 0.5)]
    # The +-2 s matching window leaves a small edge effect (0.56 here), against 1.0 unmatched.
    assert 0.5 <= surge_module.time_matched_auc(panic, normal) <= 0.6
    shifted = [(t, t + 100) for t, _ in panic]  # a real jump at the same times
    assert surge_module.time_matched_auc(shifted, normal) == 1.0


def test_smoothing_is_the_live_rolling_mean(surge_module):
    scores = np.array([0, 0, 15, 0], float)
    out = surge_module.smoothed(scores, interval_s=2.0, window_s=30.0, maxlen=15)
    np.testing.assert_allclose(out, [0, 0, 5, 3.75])
    long = surge_module.smoothed(np.arange(20, dtype=float), 2.0, 30.0, 15)
    assert long[-1] == pytest.approx(np.mean(np.arange(5, 20)))  # 15 readings within 30 s
