import time

from fastapi.testclient import TestClient

from app.main import create_app, redact
from tests.fakes import make_services


def client_for(services, run_loop=False):
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=run_loop)
    emitted = []

    async def record(event, data=None, **kwargs):
        emitted.append((event, data))

    app.sio.emit = record
    return app, TestClient(app), emitted


def test_rest_endpoints_after_manual_inference():
    services = make_services(count=700)
    app, client, emitted = client_for(services)
    with client:
        assert services.source.started
        health = client.get("/health").json()
        assert health["model_loaded"] and health["zones"] == ["RAMKUND_NORTH", "CAMERA_X"]
        assert health["storage"] == "memory" and health["cameras"]["RAMKUND_NORTH"] == {"status": "online"}

        zones = client.get("/api/zones").json()
        assert zones[0]["source"] == "rtsp://***@10.0.0.5:554/stream1"  # password hidden
        assert zones[0]["graph_node"] == "RAMKUND_NORTH" and zones[1]["graph_node"] is None

        assert client.get("/api/zones/RAMKUND_NORTH/status").json()["risk_level"] == "UNKNOWN"
        reading = client.post("/api/zones/RAMKUND_NORTH/infer").json()
        assert reading["risk_level"] == "RED" and reading["count"] > 0
        assert [name for name, _ in emitted] == ["density_update", "alert"]
        assert client.get("/api/zones/RAMKUND_NORTH/status").json() == reading

        exits = client.get("/api/zones/RAMKUND_NORTH/exits?top_k=2").json()
        assert exits["graph_node"] == "RAMKUND_NORTH" and len(exits["exits"]) == 2

        alerts = client.get("/api/alerts/recent").json()
        assert len(alerts) == 1 and alerts[0]["zone_id"] == "RAMKUND_NORTH" and alerts[0]["delivered"]
        incidents = client.get("/api/incidents").json()
        assert len(incidents) == 1 and incidents[0]["peak_level"] == "RED"
    assert services.source.stopped


def test_unknown_zone_and_off_graph_zone():
    _, client, _ = client_for(make_services())
    with client:
        assert client.get("/api/zones/NOPE/status").status_code == 404
        assert client.post("/api/zones/NOPE/infer").status_code == 404
        assert client.get("/api/zones/CAMERA_X/exits").status_code == 404


def test_runs_without_a_model():
    _, client, _ = client_for(make_services(model=False))
    with client:
        health = client.get("/health").json()
        assert not health["model_loaded"] and "not found" in health["model_error"]
        assert client.post("/api/zones/RAMKUND_NORTH/infer").status_code == 503
        assert len(client.get("/api/zones/RAMKUND_NORTH/exits").json()["exits"]) == 3


def test_background_loop_emits_density_updates():
    services = make_services(count=100, burst_interval_s=0.05)
    _, client, emitted = client_for(services, run_loop=True)
    with client:
        deadline = time.monotonic() + 5
        while sum(name == "density_update" for name, _ in emitted) < 4 and time.monotonic() < deadline:
            time.sleep(0.05)
    updates = [data for name, data in emitted if name == "density_update"]
    assert len(updates) >= 4
    assert set(updates[0]) >= {"zone_id", "density", "turbulence", "pressure", "risk_level", "count", "timestamp"}


def test_redact_leaves_plain_urls_alone():
    assert redact("rtsp://localhost:8554/pandharpur_01") == "rtsp://localhost:8554/pandharpur_01"
    assert redact("rtsp://user@cam/live") == "rtsp://***@cam/live"
