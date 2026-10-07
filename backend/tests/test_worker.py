import pytest

from app.camera_manager import DummyGrabber
from app.store import MemoryStore
from app.worker import LocalFrameSource, local_source
from tests.fakes import ZONES, make_services


def test_red_zone_emits_update_and_alert_and_is_stored():
    services = make_services(count=700)  # 700 people on 100 m² -> 7 /m², no motion -> RED
    events = services.worker.step()

    updates = {p["zone_id"]: p for name, p in events if name == "density_update"}
    assert set(updates) == {"RAMKUND_NORTH", "CAMERA_X"}
    north = updates["RAMKUND_NORTH"]
    assert north["risk_level"] == "RED" and north["density"] == pytest.approx(7.0, rel=1e-4)
    assert north["pressure"] == pytest.approx(7.0, rel=1e-4) and north["turbulence"] == 0.0
    assert len(north["exits"]) == 3 and north["exits"][0]["path"][0] == "RAMKUND_NORTH"
    assert updates["CAMERA_X"]["exits"] == []  # not on the exit graph

    alerts = [p for name, p in events if name == "alert"]
    assert {a["zone_id"] for a in alerts} == {"RAMKUND_NORTH", "CAMERA_X"}
    assert all(a["risk_level"] == "RED" and a["delivered"] for a in alerts)
    assert "Exit via" in next(a for a in alerts if a["zone_id"] == "RAMKUND_NORTH")["message"]

    assert services.router.graph.nodes["RAMKUND_NORTH"]["pressure"] == pytest.approx(7.0, rel=1e-4)
    assert len(services.store.recent_incidents()) == 2 and len(services.store.recent_alerts()) == 2


def test_calm_zone_does_not_alert():
    services = make_services(count=100)  # 1 /m²
    events = services.worker.step()
    assert [name for name, _ in events] == ["density_update", "density_update"]
    assert services.worker.latest["RAMKUND_NORTH"]["risk_level"] == "GREEN"
    assert services.worker.channel.sent == []


def test_escalation_fires_during_cooldown():
    services = make_services(count=450)  # 4.5 /m² -> ORANGE
    worker = services.worker
    assert any(name == "alert" for name, _ in worker.step())
    worker.estimator.count = 700  # smoothed (4.5 + 7) / 2 = 5.75: still ORANGE, in cooldown
    assert not any(name == "alert" for name, _ in worker.step())
    worker.estimator.count = 700  # smoothed 6.17: RED, an escalation
    alerts = [p for name, p in worker.step() if name == "alert"]
    assert alerts and all(a["risk_level"] == "RED" for a in alerts)


class BrokenStore(MemoryStore):
    def record_reading(self, status):
        raise ConnectionError("database went away")


def test_storage_errors_do_not_stop_inference():
    services = make_services(count=100, store=BrokenStore())
    events = services.worker.step()
    assert len(events) == 2 and "database went away" in services.worker.last_error


def test_infer_now_uses_the_newest_frames():
    services = make_services(count=300)
    events = services.worker.infer_now("RAMKUND_NORTH")
    assert [name for name, _ in events] == ["density_update"]
    assert services.worker.latest["RAMKUND_NORTH"]["risk_level"] == "YELLOW"


def test_local_source_maps_mediamtx_urls_to_clips(tmp_path):
    (tmp_path / "pandharpur_01.mp4").write_bytes(b"")
    assert local_source("rtsp://localhost:8554/pandharpur_01", tmp_path) == str(tmp_path / "pandharpur_01.mp4")
    assert local_source("rtsp://localhost:8554/missing", tmp_path) == "rtsp://localhost:8554/missing"
    assert local_source("rtsp://10.0.0.5/stream", tmp_path) == "rtsp://10.0.0.5/stream"


class StalledGrabber(DummyGrabber):
    def latest_burst(self):
        return [], 0.0, 1  # the frame number never advances


def test_local_frame_source_returns_each_burst_once():
    source = LocalFrameSource(ZONES[:1], grabber_factory=StalledGrabber)
    source.start()
    assert source.next_burst("RAMKUND_NORTH") is not None
    assert source.next_burst("RAMKUND_NORTH") is None  # same frame number: nothing new
    assert source.peek_burst("RAMKUND_NORTH") is not None
    assert source.camera_status("RAMKUND_NORTH")["status"] == "online"
    source.stop()
