"""Optional speed-surge trigger (app.surge): the rule, the density gate, the cooldown, and default off."""

import pytest

import app.pipeline as pipeline
from app.config import get_settings
from app.surge import SurgeTracker
from ml.optical_flow import TurbulenceResult
from tests.fakes import FakeSource, make_services
from tests.test_api import client_for


def test_tracker_ratio_needs_history_and_uses_the_recent_median():
    t = SurgeTracker(ratio=1.10, consecutive=2, history_s=10)
    first = t.update(1.0, 0.0)
    assert first.ratio is None and first.streak == 0 and not first.fired
    for k in range(1, 6):
        r = t.update(1.0, 2.0 * k)
    assert r.ratio == pytest.approx(1.0) and r.streak == 0
    r = t.update(2.0, 12.0)  # median of 2.0 .. 10.0 s (5 readings) = 1.0
    assert r.ratio == pytest.approx(2.0) and r.streak == 1 and not r.fired
    r = t.update(2.0, 14.0)  # median of 4.0 .. 12.0 s = 1.0: second reading in a row
    assert r.ratio == pytest.approx(2.0) and r.streak == 2 and r.fired
    r = t.update(1.0, 16.0)
    assert r.streak == 0 and not r.fired


def test_tracker_single_spike_does_not_fire_and_floor_applies():
    t = SurgeTracker(ratio=1.10, consecutive=2, history_s=10)
    for k, speed in enumerate([1.0, 1.0, 1.0, 3.0, 1.0, 3.0, 1.0]):
        assert not t.update(speed, 2.0 * k).fired  # never two in a row
    still = SurgeTracker(history_s=10)
    still.update(0.0, 0.0)
    assert still.update(1.0, 2.0).ratio == pytest.approx(1.0 / 0.5)  # empty scene: median floored at 0.5 px/frame
    old = SurgeTracker(history_s=10)
    old.update(9.0, 0.0)
    old.update(1.0, 11.0)
    assert old.update(1.0, 12.0).ratio == pytest.approx(1.0)  # the 9.0 at t=0 is older than 10 s
    old.reset()
    assert old.update(1.0, 14.0).ratio is None
    with pytest.raises(ValueError):
        SurgeTracker(ratio=0)


class TimedSource(FakeSource):
    """The motionless test burst, stamped `interval_s` later each time a zone asks for one."""

    def __init__(self, interval_s: float = 2.0):
        super().__init__()
        self.interval_s, self.clock = interval_s, {}

    def next_burst(self, zone_id):
        self.clock[zone_id] = self.clock.get(zone_id, 1_000.0) + self.interval_s
        return self.frames, self.clock[zone_id]


@pytest.fixture
def speeds(monkeypatch):
    """Set `speeds.value` to the crowd speed the next readings report (turbulence stays 0)."""
    class Speed:
        value = 1.0

    def fake_turbulence(frames, *args, **kwargs):
        return TurbulenceResult(score=0.0, mean_speed=Speed.value, moving_fraction=0.5, valid_blocks=8, pairs_used=4)

    monkeypatch.setattr(pipeline, "burst_turbulence", fake_turbulence)
    return Speed


def run(services, speeds, pattern):
    """One worker step per speed in `pattern`; the north zone's (payload, alerts) per step."""
    out = []
    for value in pattern:
        speeds.value = value
        events = services.worker.step()
        payload = next(p for name, p in events if name == "density_update" and p["zone_id"] == "RAMKUND_NORTH")
        alerts = [p for name, p in events if name == "alert" and p["zone_id"] == "RAMKUND_NORTH"]
        out.append((payload, alerts))
    return out


CALM, SURGE = [1.0] * 5, [3.0, 3.0]


def services_with(**overrides):
    """make_services with the surge settings pinned (whatever .env says), overridable per test."""
    pinned = {"surge_ratio": 1.10, "surge_consecutive": 2, "surge_history_s": 10.0, "surge_min_level": "YELLOW",
              "alert_cooldown_s": 600.0}
    return make_services(source=TimedSource(), **{**pinned, **overrides})


def test_default_off_changes_nothing_but_shows_the_ratio(speeds):
    services = services_with(count=250, surge_trigger=False)  # 2.5 /m2: YELLOW
    steps = run(services, speeds, CALM + SURGE)
    assert [p["risk_level"] for p, _ in steps] == ["YELLOW"] * 7
    assert not any(alerts for _, alerts in steps) and services.worker.channel.sent == []
    last = steps[-1][0]
    assert last["surge_ratio"] == pytest.approx(3.0) and last["surge_fired"] and last["escalation"] is None
    assert steps[0][0]["surge_ratio"] is None  # no history yet


def test_two_readings_in_a_row_raise_a_yellow_zone_to_orange_with_the_reason(speeds):
    services = services_with(count=250, surge_trigger=True)
    steps = run(services, speeds, CALM + SURGE)
    assert [p["risk_level"] for p, _ in steps] == ["YELLOW"] * 6 + ["ORANGE"]  # not on the first surge reading
    payload, alerts = steps[-1]
    assert payload["escalation"] == "speed surge" and payload["pressure"] == pytest.approx(2.5, rel=1e-4)
    assert len(alerts) == 1 and alerts[0]["risk_level"] == "ORANGE" and alerts[0]["reason"] == "speed surge"
    assert "SPEED SURGE" in alerts[0]["message"] and "3.0x" in alerts[0]["message"]
    assert services.store.recent_alerts()[0]["reason"] == "speed surge"
    assert services.store.recent_incidents()[0]["peak_level"] == "ORANGE"


def test_a_single_spike_does_not_escalate(speeds):
    services = services_with(count=250, surge_trigger=True)
    steps = run(services, speeds, CALM + [3.0, 1.0, 3.0, 1.0])
    assert all(p["risk_level"] == "YELLOW" for p, _ in steps) and not any(a for _, a in steps)


def test_density_gate_keeps_a_green_zone_green(speeds):
    services = services_with(count=100, surge_trigger=True)  # 1 /m2: GREEN
    steps = run(services, speeds, CALM + SURGE)
    assert all(p["risk_level"] == "GREEN" for p, _ in steps) and steps[-1][0]["surge_fired"]
    assert services.worker.channel.sent == []
    ungated = services_with(count=100, surge_trigger=True, surge_min_level="GREEN")
    assert run(ungated, speeds, CALM + SURGE)[-1][0]["risk_level"] == "ORANGE"


def test_a_zone_already_orange_or_red_is_left_as_it_is(speeds):
    services = services_with(count=700, surge_trigger=True)  # RED
    steps = run(services, speeds, CALM + SURGE)
    assert steps[-1][0]["risk_level"] == "RED" and steps[-1][0]["escalation"] is None


def test_cooldown_is_respected(speeds):
    services = services_with(count=250, surge_trigger=True, alert_cooldown_s=60)
    first = run(services, speeds, CALM + SURGE)
    assert sum(len(a) for _, a in first) == 1  # the escalation to ORANGE
    again = run(services, speeds, CALM + SURGE)  # ~14 s later: back to YELLOW, then ORANGE again
    assert again[-1][0]["risk_level"] == "ORANGE" and not any(a for _, a in again)  # within the cooldown
    later = run(services, speeds, CALM * 5 + SURGE)  # > 60 s after the alert
    assert later[-1][0]["risk_level"] == "ORANGE" and len(later[-1][1]) == 1  # the reminder


def test_scene_cut_forgets_the_history(speeds, monkeypatch):
    services = services_with(count=250, surge_trigger=True)
    run(services, speeds, CALM)
    monkeypatch.setattr(pipeline, "burst_has_cut", lambda frames, params: True)
    services.worker.step()  # a cut: no reading
    monkeypatch.setattr(pipeline, "burst_has_cut", lambda frames, params: False)
    payload, _ = run(services, speeds, [3.0])[0]
    assert payload["surge_ratio"] is None  # speeds before the cut are not compared with


def test_settings_and_config_endpoint(monkeypatch):
    with pytest.raises(ValueError, match="SURGE_MIN_LEVEL"):
        make_services(surge_min_level="AMBER")
    for name, value in (("SURGE_TRIGGER", "1"), ("SURGE_RATIO", "1.3"), ("SURGE_CONSECUTIVE", "3"),
                        ("SURGE_HISTORY_S", "12"), ("SURGE_MIN_LEVEL", "orange")):
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert (s.surge_trigger, s.surge_ratio, s.surge_consecutive, s.surge_history_s, s.surge_min_level) == \
            (True, 1.3, 3, 12.0, "ORANGE")
    finally:
        monkeypatch.undo()  # the endpoint check below uses the defaults again
        get_settings.cache_clear()
    _, client, _ = client_for(make_services(surge_trigger=False))
    with client:
        surge = client.get("/api/config").json()["surge"]
    assert surge == {"trigger": False, "ratio": 1.10, "consecutive": 2, "history_s": 10.0, "min_level": "YELLOW"}
