import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.config import redact
from app.main import create_app
from tests.fakes import FakeSource, make_services, real_estimator


def client_for(services, run_loop=False):
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=run_loop)
    emitted = []

    async def record(event, data=None, **kwargs):
        emitted.append((event, data))

    app.sio.emit = record
    return app, TestClient(app), emitted


def decode(response, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.frombuffer(response.content, np.uint8), flags)


def test_rest_endpoints_after_manual_inference():
    services = make_services(count=700)
    app, client, emitted = client_for(services)
    with client:
        assert services.source.started
        health = client.get("/health").json()
        assert health["model_loaded"] and health["zones"] == ["RAMKUND_NORTH", "CAMERA_X"]
        assert "model" in health  # which weights are active (None for the fake estimator)
        assert health["storage"] == "memory" and health["cameras"]["RAMKUND_NORTH"] == {"status": "online"}

        zones = client.get("/api/zones").json()
        assert zones[0]["source"] == "rtsp://***@10.0.0.5:554/stream1"  # password hidden
        assert zones[0]["graph_node"] == "RAMKUND_NORTH" and zones[1]["graph_node"] is None

        before = client.get("/api/zones/RAMKUND_NORTH/status").json()
        assert before["risk_level"] == "UNKNOWN" and "server_time" in before
        reading = client.post("/api/zones/RAMKUND_NORTH/infer").json()
        assert reading["risk_level"] == "RED" and reading["count"] > 0
        assert [name for name, _ in emitted] == ["density_update", "alert"]
        assert abs(emitted[0][1]["server_time"] - time.time()) < 5
        status = client.get("/api/zones/RAMKUND_NORTH/status").json()
        assert status["risk_level"] == "RED" and status["timestamp"] == reading["timestamp"]

        exits = client.get("/api/zones/RAMKUND_NORTH/exits?top_k=2").json()
        assert exits["graph_node"] == "RAMKUND_NORTH" and len(exits["exits"]) == 2

        [alert] = client.get("/api/alerts/recent").json()
        assert alert == emitted[1][1]  # REST and Socket.IO share one shape
        assert alert["zone_id"] == "RAMKUND_NORTH" and alert["delivered"] and alert["deliveries"][0]["channel"] == "fake"
        incidents = client.get("/api/incidents").json()
        assert len(incidents) == 1 and incidents[0]["peak_level"] == "RED"
    assert services.source.stopped


def test_config_exposes_thresholds():
    _, client, _ = client_for(make_services(risk_yellow=1.5, risk_orange=3.5, risk_red=5.5))
    with client:
        config = client.get("/api/config").json()
    assert config["thresholds"] == {"yellow": 1.5, "orange": 3.5, "red": 5.5}
    assert config["burst_interval_s"] > 0


def test_history_returns_recent_readings_oldest_first():
    services = make_services(count=300)
    _, client, _ = client_for(services)
    with client:
        for _ in range(4):
            services.worker.step()
        body = client.get("/api/zones/RAMKUND_NORTH/history?limit=3").json()
        assert body["zone_id"] == "RAMKUND_NORTH" and "server_time" in body
        readings = body["readings"]
        assert len(readings) == 3 and [r["timestamp"] for r in readings] == sorted(r["timestamp"] for r in readings)
        assert set(readings[0]) >= {"timestamp", "pressure", "raw_pressure", "density", "count", "risk_level"}
        assert readings[-1]["pressure"] == client.get("/api/zones/RAMKUND_NORTH/status").json()["pressure"]
        assert client.get("/api/zones/NOPE/history").status_code == 404


def test_snapshot_and_density_overlay_line_up():
    services = make_services(count=300)
    _, client, _ = client_for(services)
    with client:
        services.worker.step()
        snap = client.get("/api/zones/RAMKUND_NORTH/snapshot.jpg")
        assert snap.status_code == 200 and snap.headers["content-type"] == "image/jpeg"
        assert snap.headers["cache-control"] == "no-store" and float(snap.headers["x-frame-timestamp"]) > 0
        density = client.get("/api/zones/RAMKUND_NORTH/density.png")
        assert density.status_code == 200 and density.headers["content-type"] == "image/png"
        snapshot_image, overlay = decode(snap), decode(density)
        assert overlay.shape[2] == 4  # BGRA: transparent where there is no crowd
        assert overlay.shape[:2] == snapshot_image.shape[:2] == (64, 96)
        # 300 people on 100 m² = 3 /m²: between yellow (2) and orange (4) on the fixed scale
        b, g, r, alpha = overlay[32, 48].astype(int)
        assert r > g > b and 150 < g < 215 and 0.4 * 255 < alpha < 0.75 * 255
        scale = client.get("/api/config").json()["density_scale"]
        assert scale["unit"] == "persons/m²" and [s["value"] for s in scale["stops"]] == [0, 2, 4, 6, 7]


def test_large_frames_are_scaled_to_display_width():
    frames = [np.zeros((1080, 1920, 3), np.uint8)] * 5
    services = make_services(count=300, source=FakeSource(frames))
    _, client, _ = client_for(services)
    with client:
        services.worker.step()
        assert decode(client.get("/api/zones/RAMKUND_NORTH/snapshot.jpg")).shape[:2] == (540, 960)
        assert decode(client.get("/api/zones/RAMKUND_NORTH/density.png")).shape[:2] == (540, 960)


def test_gradcam_overlay(tmp_path):
    services = make_services(estimator=real_estimator(tmp_path))
    _, client, _ = client_for(services)
    with client:
        assert client.get("/api/zones/RAMKUND_NORTH/gradcam.png").status_code == 503  # not analysed yet
        services.worker.step()
        first = client.get("/api/zones/RAMKUND_NORTH/gradcam.png")
        assert first.status_code == 200 and first.headers["content-type"] == "image/png"
        image = decode(first)
        assert image.shape == (64, 96, 4)
        assert client.get("/api/zones/RAMKUND_NORTH/gradcam.png").content == first.content  # cached per frame


def test_images_need_a_known_zone_a_frame_and_a_model():
    _, client, _ = client_for(make_services(source=FakeSource(empty=True)))
    with client:
        for path in ("snapshot.jpg", "density.png", "gradcam.png", "history"):
            assert client.get(f"/api/zones/NOPE/{path}").status_code == 404
        response = client.get("/api/zones/RAMKUND_NORTH/snapshot.jpg")
        assert response.status_code == 503 and "no frame" in response.json()["detail"]
        assert client.get("/api/zones/RAMKUND_NORTH/density.png").status_code == 503

    _, client, _ = client_for(make_services(model=False))
    with client:
        assert client.get("/api/zones/RAMKUND_NORTH/snapshot.jpg").status_code == 200  # camera works
        for path in ("density.png", "gradcam.png"):
            response = client.get(f"/api/zones/RAMKUND_NORTH/{path}")
            assert response.status_code == 503 and "model not loaded" in response.json()["detail"]


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
    assert set(updates[0]) >= {"zone_id", "density", "turbulence", "pressure", "risk_level", "count", "timestamp",
                               "server_time"}


def test_cors_exposes_frame_timestamp():
    _, client, _ = client_for(make_services())
    with client:
        response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "*"
    assert "x-frame-timestamp" in response.headers["access-control-expose-headers"].lower()


def test_redact_leaves_plain_urls_alone():
    assert redact("rtsp://localhost:8554/pandharpur_01") == "rtsp://localhost:8554/pandharpur_01"
    assert redact("rtsp://user@cam/live") == "rtsp://***@cam/live"


class RecordingLoop:
    def __init__(self) -> None:
        self.forwarded = []

    def default_exception_handler(self, context):
        self.forwarded.append(context)


def test_a_dropped_browser_connection_is_not_logged_as_an_error():
    from app.main import quiet_connection_resets

    loop = RecordingLoop()
    quiet_connection_resets(loop, {"message": "Exception in callback _call_connection_lost",
                                   "exception": ConnectionResetError(10054, "forcibly closed")})
    assert loop.forwarded == []  # Windows proactor noise: dropped
    real = {"message": "Task exception was never retrieved", "exception": ValueError("bug")}
    quiet_connection_resets(loop, real)
    assert loop.forwarded == [real]  # anything else still reaches the default handler
