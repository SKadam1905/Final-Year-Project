"""P5 reports: filtered alert/incident history (both stores), per-zone summaries and the endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.reports import summarise_readings
from app.store import MemoryStore
from tests.fakes import auth_headers, make_services
from tests.test_store import exercise, mongo_store


@pytest.fixture(params=["memory", "mongodb"])
def store(request):
    """Readings at t = 0..10 s, incident 1 (t 2-8, RED), incident 2 (t 10-, open), RED alerts at t 4 and 10."""
    s = MemoryStore() if request.param == "memory" else mongo_store()
    exercise(s)
    yield s
    s.close()


def test_alert_history_filters(store):
    assert [a["timestamp"] for a in store.alert_history(levels=["RED"])] == [10.0, 4.0]
    assert store.alert_history(levels=["ORANGE"]) == []
    assert [a["timestamp"] for a in store.alert_history(since=5)] == [10.0]
    assert [a["timestamp"] for a in store.alert_history(until=5)] == [4.0]
    assert store.alert_history(zone_id="other") == [] and len(store.alert_history(zone_id="Z", limit=1)) == 1


def test_incident_history_keeps_incidents_overlapping_the_range(store):
    assert [i["ended_at"] for i in store.incident_history(since=9)] == [None]  # the first ended at t=8
    assert len(store.incident_history(until=3)) == 1  # the second started at t=10
    assert len(store.incident_history(levels=["RED"])) == 2 and store.incident_history(levels=["YELLOW"]) == []


def test_readings_between(store):
    readings = store.readings_between(3, 9)
    assert [r["timestamp"] for r in readings] == [4.0, 6.0, 8.0]
    assert readings[0]["zone_id"] == "Z" and readings[0]["risk_level"] == "RED"
    assert store.readings_between(zone_id="other") == []


def test_summary_peaks_and_time_at_each_level():
    readings = [
        {"zone_id": "A", "timestamp": 0, "count": 10, "density": 1.0, "pressure": 1.0, "risk_level": "GREEN"},
        {"zone_id": "A", "timestamp": 2, "count": 50, "density": 4.5, "pressure": 4.8, "risk_level": "ORANGE"},
        {"zone_id": "A", "timestamp": 4, "count": 40, "density": 6.5, "pressure": 6.6, "risk_level": "RED",
         "count_method": "detection"},
        {"zone_id": "A", "timestamp": 30, "count": 5, "density": 0.5, "pressure": 3.0, "risk_level": "GREEN"},
        {"zone_id": "B", "timestamp": 1, "count": 3, "density": 0.1, "pressure": 0.1, "risk_level": "GREEN"},
    ]
    a, b = summarise_readings(readings, interval_s=2, max_gap_s=10)
    assert (a["zone_id"], a["readings"], a["first"], a["last"]) == ("A", 4, 0, 30)
    assert (a["peak_count"], a["peak_count_time"], a["peak_density"], a["peak_pressure"]) == (50, 2, 6.5, 6.6)
    # RED at t=4 holds until t=30, capped at 10 s (a gap that long is an outage); the last reading holds 2 s.
    assert a["time_at_level_s"] == {"GREEN": 4.0, "YELLOW": 0.0, "ORANGE": 2.0, "RED": 10.0}
    assert a["highest_level"] == "RED" and a["detection_readings"] == 1
    assert b["time_at_level_s"]["GREEN"] == 2.0 and b["highest_level"] == "GREEN"


def test_report_endpoints():
    services = make_services(count=700)  # 7 /m²: RED in both zones
    services.worker.step()
    services.worker.estimator.count = 100  # 1 /m²: the 30 s mean stays high, still RED
    services.worker.step()
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=False)
    with TestClient(app, headers=auth_headers(services.settings)) as client:
        alerts = client.get("/api/reports/alerts", params={"level": "red"}).json()
        assert {a["zone_id"] for a in alerts} == {"RAMKUND_NORTH", "CAMERA_X"}
        assert client.get("/api/reports/alerts", params={"zone_id": "CAMERA_X"}).json()[0]["zone_id"] == "CAMERA_X"
        assert client.get("/api/reports/alerts", params={"since": 4e9}).json() == []
        bad = client.get("/api/reports/alerts", params={"level": "PURPLE"})
        assert bad.status_code == 400 and "PURPLE" in bad.json()["detail"]

        assert len(client.get("/api/reports/incidents").json()) == 2

        summary = client.get("/api/reports/summary").json()
        assert summary["storage"] == "memory" and summary["readings"] == 4
        north = next(z for z in summary["zones"] if z["zone_id"] == "RAMKUND_NORTH")
        assert north["name"] == "Ramkund north steps" and north["readings"] == 2
        assert north["highest_level"] == "RED" and north["peak_count"] == pytest.approx(700, rel=1e-4)
