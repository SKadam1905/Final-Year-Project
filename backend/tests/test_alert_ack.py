"""P5: acknowledging alerts (POST /api/alerts/{id}/ack) and the structured alert fields."""

import pytest
from fastapi.testclient import TestClient

from app.alert_dispatcher import AlertEvent, Delivery
from app.main import create_app
from app.risk_engine import RiskLevel
from app.store import MemoryStore
from tests.fakes import auth_headers, make_services
from tests.test_store import mongo_store

RED = RiskLevel.RED
HOTSPOT = {"row": 0, "col": 2, "label": "top-right", "count": 90.0, "density": 6.1}


def event(t=10.0):
    return AlertEvent("Z", RED, "RED: Zone Z", t, (Delivery("console", "stdout", True, 0.0),), ("Z", "EXIT_GATE_A"),
                      HOTSPOT, ("STOP inflow at all gates feeding this zone: Gate C.", "Open all exits."))


@pytest.fixture(params=["memory", "mongodb"])
def store(request):
    s = MemoryStore() if request.param == "memory" else mongo_store()
    yield s
    s.close()


def test_alerts_carry_the_hotspot_and_gate_actions(store):
    stored = store.record_alert(event())
    store.flush()
    assert stored["hotspot"] == HOTSPOT and stored["gate_actions"][0].startswith("STOP inflow")
    assert stored["acknowledged_by"] is None and stored["acknowledged_at"] is None
    assert store.recent_alerts()[0]["gate_actions"] == stored["gate_actions"]


def test_acknowledging_records_who_and_when_and_the_first_one_stands(store):
    stored = store.record_alert(event())
    acked = store.acknowledge(str(stored["id"]), "officer", now=100.0)
    assert (acked["acknowledged_by"], acked["acknowledged_at"]) == ("officer", 100.0)
    again = store.acknowledge(str(stored["id"]), "admin", now=200.0)
    assert (again["acknowledged_by"], again["acknowledged_at"]) == ("officer", 100.0)
    store.flush()
    reloaded = store.recent_alerts()[0]
    assert (reloaded["acknowledged_by"], reloaded["acknowledged_at"]) == ("officer", 100.0)
    assert store.acknowledge("12345", "officer") is None and store.acknowledge("not-an-id", "officer") is None


def test_mongo_acknowledges_an_alert_not_in_its_cache():
    client_store = mongo_store()
    stored = client_store.record_alert(event())
    client_store.flush()
    client_store._alert_cache.clear()  # e.g. after a backend restart
    acked = client_store.acknowledge(stored["id"], "officer", now=50.0)
    client_store.flush()
    assert acked["acknowledged_by"] == "officer" and client_store.recent_alerts()[0]["acknowledged_at"] == 50.0
    client_store.close()


def test_ack_endpoint_for_an_officer_emits_alert_ack():
    services = make_services(count=700)  # RED: alerts for both zones
    services.worker.step()
    alert = services.store.recent_alerts()[0]
    assert alert["hotspot"] is not None and alert["gate_actions"]  # from the live pipeline
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=False)
    emitted = []

    async def record(name, data=None, **kwargs):
        emitted.append((name, data))

    app.sio.emit = record
    with TestClient(app, headers=auth_headers(services.settings, role="officer")) as client:
        response = client.post(f"/api/alerts/{alert['id']}/ack")
        assert response.status_code == 200
        body = response.json()
        assert body["acknowledged_by"] == "test-officer" and body["acknowledged_at"] > 0
        assert emitted == [("alert_ack", body)]
        assert client.get("/api/alerts/recent").json()[0]["acknowledged_by"] == "test-officer"
        missing = client.post("/api/alerts/99999/ack")
        assert missing.status_code == 404 and "no alert" in missing.json()["detail"]
    with TestClient(app) as anonymous:
        assert anonymous.post(f"/api/alerts/{alert['id']}/ack").status_code == 401
