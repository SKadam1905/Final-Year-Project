"""Data made with demo or provenance-less weights is hidden unless DEMO_MODE=1, and can be moved out."""

from datetime import datetime, timezone

import mongomock
from fastapi.testclient import TestClient

from app.alert_dispatcher import AlertEvent, Delivery
from app.main import create_app
from app.risk_engine import RiskLevel, ZoneStatus
from tools.quarantine_demo_data import demo_query, move, summarise
from tests.fakes import auth_headers, make_services
from tests.test_store import mongo_store

RED, GREEN = RiskLevel.RED, RiskLevel.GREEN


def reading(zone, level, t, count=100.0):
    return ZoneStatus(zone, t, count, 7.0, 0.1, 7.0, 7.0, level, 15)


def fill(store):
    """Zone D: RED readings and an alert from demo weights; zone R: the same from real weights."""
    for kind, zone in (("demo", "D"), ("public_pretrained", "R")):
        store.model_kind = kind
        store.record_reading(reading(zone, RED, 10.0, count=700.05 if kind == "demo" else 120.0))
        store.record_alert(AlertEvent(zone, RED, f"RED: {zone}", 10.0, (Delivery("console", "stdout", True, 0.0),)))
    store.flush()


def test_demo_data_is_hidden_unless_demo_mode():
    client = mongomock.MongoClient(tz_aware=True)
    store = mongo_store(client)
    fill(store)
    assert [a["zone_id"] for a in store.recent_alerts()] == ["R"]
    assert [a["zone_id"] for a in store.alert_history()] == ["R"]
    assert [i["zone_id"] for i in store.recent_incidents()] == ["R"]
    assert [i["zone_id"] for i in store.incident_history()] == ["R"]
    assert store.history("D") == [] and len(store.history("R")) == 1
    assert [r["zone_id"] for r in store.readings_between()] == ["R"]
    assert "model_kind" not in store.history("R")[0]  # same shape as the in-memory store

    shown = mongo_store(client, show_demo=True)  # DEMO_MODE=1
    assert sorted(a["zone_id"] for a in shown.recent_alerts()) == ["D", "R"]
    assert len(shown.history("D")) == 1
    assert set(shown._open) == {"D", "R"} and set(mongo_store(client)._open) == {"R"}


def test_the_api_never_returns_demo_alerts_or_readings():
    store = mongo_store()
    fill(store)
    services = make_services(store=store)
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=False)
    with TestClient(app, headers=auth_headers(services.settings)) as http:
        assert [a["zone_id"] for a in http.get("/api/alerts/recent").json()] == ["R"]
        assert [a["zone_id"] for a in http.get("/api/reports/alerts").json()] == ["R"]
        assert http.get("/api/zones/RAMKUND_NORTH/history").json()["readings"] == []
        assert [z["zone_id"] for z in http.get("/api/reports/summary").json()["zones"]] == ["R"]


def test_quarantine_moves_tagged_and_old_untagged_demo_data_to_a_backup():
    client = mongomock.MongoClient(tz_aware=True)
    db = client["kumbh_test"]
    t = lambda h: datetime(2026, 10, 8, h, tzinfo=timezone.utc)  # noqa: E731
    db.zone_readings.insert_many([
        {"zone_id": "A", "ts": t(14), "count": 700.05},  # untagged, before the cut-off: the old demo model
        {"zone_id": "A", "ts": t(23), "count": 120.0},  # untagged, after: real
        {"zone_id": "A", "ts": t(23), "count": 700.05, "model_kind": "demo"},  # tagged demo
        {"zone_id": "A", "ts": t(23), "count": 130.0, "model_kind": "public_pretrained"},
    ])
    db.alerts.insert_many([{"zone_id": "A", "triggered_at": t(14), "message": "~700 people"},
                           {"zone_id": "A", "triggered_at": t(23), "message": "~120 people"}])
    cutoff = t(20)

    plan = summarise(db, cutoff)
    assert plan["zone_readings"]["count"] == 2 and plan["zone_readings"]["kept"] == 2
    assert plan["zone_readings"]["untagged_not_demo_count"] == 0  # every untagged one before the cut-off counts 700.05
    assert plan["alerts"]["count"] == 1 and plan["incidents"]["count"] == 0
    assert db.zone_readings.count_documents({}) == 4  # summarise moves nothing

    assert move(db, "zone_readings", demo_query("ts", cutoff), batch=1) == 2
    assert move(db, "alerts", demo_query("triggered_at", cutoff)) == 1
    assert sorted(d["count"] for d in db.zone_readings.find()) == [120.0, 130.0]
    assert sorted(d["count"] for d in db.zone_readings_demo_backup.find()) == [700.05, 700.05]
    assert [d["message"] for d in db.alerts.find()] == ["~120 people"]
    assert move(db, "zone_readings", demo_query("ts", cutoff)) == 0  # running it again changes nothing
