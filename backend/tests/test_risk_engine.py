import json

import pytest

from app.risk_engine import (
    RiskEngine,
    RiskLevel,
    RiskThresholds,
    RollingWindow,
    ZoneStatus,
    classify_risk,
    crowd_pressure,
)


def test_crowd_pressure_formula():
    assert crowd_pressure(5.0, 0.6, k=0.5) == pytest.approx(6.5)  # 5.0 * (1 + 0.5 * 0.6)
    assert crowd_pressure(4.0, 0.0) == 4.0
    assert crowd_pressure(4.0, 1.0, k=0.5) == pytest.approx(6.0)


def test_turbulence_is_clipped_to_unit_range():
    assert crowd_pressure(2.0, 5.0, k=0.5) == pytest.approx(3.0)
    assert crowd_pressure(2.0, -1.0, k=0.5) == pytest.approx(2.0)


def test_negative_or_nan_density_rejected():
    with pytest.raises(ValueError):
        crowd_pressure(-0.1, 0.0)
    with pytest.raises(ValueError):
        crowd_pressure(float("nan"), 0.0)


def test_turbulent_crowd_scores_above_a_denser_static_one():
    # Plan Q2: 3/m² with high turbulence is riskier than a stationary 4/m².
    assert crowd_pressure(3.0, 1.0) > crowd_pressure(4.0, 0.0)


@pytest.mark.parametrize(
    "pressure, level",
    [(0.0, "GREEN"), (1.9, "GREEN"), (1.99, "GREEN"), (2.0, "YELLOW"), (3.99, "YELLOW"),
     (4.0, "ORANGE"), (5.99, "ORANGE"), (6.0, "RED"), (12.0, "RED")],
)
def test_classify_risk_boundaries(pressure, level):
    assert classify_risk(pressure) == level
    assert classify_risk(pressure) is RiskLevel(level)


def test_levels_are_strings_ordered_by_severity():
    assert str(RiskLevel.RED) == "RED" and RiskLevel.RED == "RED"
    assert RiskLevel.GREEN < RiskLevel.YELLOW < RiskLevel.ORANGE < RiskLevel.RED
    assert RiskLevel.RED > "ORANGE" and max(RiskLevel.ORANGE, RiskLevel.YELLOW) is RiskLevel.ORANGE
    assert json.dumps({"level": RiskLevel.ORANGE}) == '{"level": "ORANGE"}'


def test_thresholds_must_increase():
    with pytest.raises(ValueError):
        RiskThresholds(yellow=4.0, orange=2.0, red=6.0)


def test_rolling_window_expires_by_time_and_count():
    window = RollingWindow(window_s=30.0, maxlen=15)
    for t in range(0, 30, 2):
        window.add(t, 10.0)
    assert len(window) == 15
    window.add(40.0, 0.0)  # samples at t <= 10 are now older than 30 s
    assert len(window) == 10
    assert window.mean() == pytest.approx(90.0 / 10)


def test_rolling_window_empty_mean_raises():
    with pytest.raises(ValueError):
        RollingWindow().mean()


def test_engine_smooths_over_the_last_15_readings():
    engine = RiskEngine("z1", window=15)
    for density in range(1, 16):  # densities 1..15, no turbulence
        status = engine.update(float(density), 0.0)
    assert status["smoothed_pressure"] == pytest.approx(8.0)  # mean of 1..15
    status = engine.update(16.0, 0.0)  # the oldest reading (1) drops out
    assert status["smoothed_pressure"] == pytest.approx(9.0)  # mean of 2..16
    assert status["raw_pressure"] == 16.0 and status["samples"] == 15
    assert set(status) == {"zone_id", "raw_pressure", "smoothed_pressure", "risk_level", "density",
                           "turbulence", "samples"}


def test_all_green_readings_stay_green():
    engine = RiskEngine("z1")
    assert all(engine.update(1.0, 0.2)["risk_level"] == "GREEN" for _ in range(30))


def test_all_red_readings_stay_red():
    engine = RiskEngine("z1")
    assert all(engine.update(7.0, 0.5)["risk_level"] == "RED" for _ in range(30))


def test_single_frame_spike_is_suppressed():
    engine = RiskEngine("z1")
    for i in range(14):
        engine.update(1.0, 0.0, timestamp=i * 2.0)
    status = engine.update(10.0, 0.0, timestamp=28.0)
    assert status["raw_pressure"] == pytest.approx(10.0)
    assert status["smoothed_pressure"] == pytest.approx((14 * 1.0 + 10.0) / 15)
    assert status["risk_level"] == "GREEN"


def test_sustained_density_escalates():
    engine = RiskEngine("z1")
    levels = [engine.update(5.0, 0.0, timestamp=i * 2.0)["risk_level"] for i in range(15)]
    assert levels[-1] == "ORANGE"


def test_outage_does_not_keep_stale_readings():
    engine = RiskEngine("z1")
    for i in range(15):
        engine.update(8.0, 0.5, timestamp=i * 2.0)
    status = engine.update(0.5, 0.0, timestamp=120.0)
    assert status["samples"] == 1 and status["risk_level"] == "GREEN"


def test_get_status_and_reset():
    engine = RiskEngine("z1", thresholds=RiskThresholds(1.0, 2.0, 3.0))
    assert engine.get_status() is None
    status = engine.update(2.5, 0.0)
    assert engine.get_status() == status and status["risk_level"] == "ORANGE"
    engine.reset()
    assert engine.get_status() is None
    assert engine.update(0.5, 0.0)["samples"] == 1


def test_zone_status_to_dict_is_json_friendly():
    status = ZoneStatus("z1", 0.0, 10.0, 1.0, 0.1, 1.05, 1.05, RiskLevel.GREEN, 1)
    data = status.to_dict()
    assert data["level"] == "GREEN" and type(data["level"]) is str
    json.dumps(data)
