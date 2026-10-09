import logging
import time
from collections import deque

import numpy as np
import pytest

from app.camera_manager import DummyGrabber, FrameGrabber
from app.config import ZoneConfig
from app.store import MemoryStore
from app.worker import FrameSourceUnavailable, LocalFrameSource, RedisFrameSource, local_source
from tests.conftest import texture
from tests.fakes import ZONES, FakeSource, make_services


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
    assert all(a["risk_level"] == "RED" and a["delivered"] and a["id"] for a in alerts)
    assert "Exit via" in next(a for a in alerts if a["zone_id"] == "RAMKUND_NORTH")["message"]

    assert services.router.graph.nodes["RAMKUND_NORTH"]["pressure"] == pytest.approx(7.0, rel=1e-4)
    assert len(services.store.recent_incidents()) == 2
    assert services.store.recent_alerts() == alerts[::-1]  # the emitted events are what REST serves

    frame = services.worker.frames["RAMKUND_NORTH"]
    assert frame.frame.shape == (64, 96, 3) and frame.density.sum() == pytest.approx(700, rel=1e-4)
    assert frame.timestamp == north["timestamp"]


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


def test_a_burst_spanning_a_scene_cut_gives_no_reading():
    a = np.dstack([texture(64, 96, seed=1)] * 3)
    b = np.dstack([texture(64, 96, seed=2)] * 3) // 2 + 120  # another shot: different grey levels
    services = make_services(count=700, source=FakeSource([a, a, b, b, b]))
    assert services.worker.step() == []  # no update, no alert
    assert services.worker.analyzers["RAMKUND_NORTH"].engine.get_status() is None  # risk mean untouched
    assert services.store.readings == deque()


class BrokenStore(MemoryStore):
    def record_reading(self, status):
        raise ConnectionError("database went away")


def test_storage_errors_do_not_stop_inference():
    services = make_services(count=100, store=BrokenStore())
    events = services.worker.step()
    assert len(events) == 2 and "database went away" in services.worker.last_error


class NoAlertStorage(MemoryStore):
    def record_alert(self, event):
        raise ConnectionError("database went away")


def test_alert_is_still_emitted_when_it_cannot_be_stored():
    services = make_services(count=700, store=NoAlertStorage())
    alerts = [p for name, p in services.worker.step() if name == "alert"]
    assert len(alerts) == 2 and all(a["id"] is None and a["risk_level"] == "RED" for a in alerts)


def test_infer_now_uses_the_newest_frames():
    services = make_services(count=300)
    events = services.worker.infer_now("RAMKUND_NORTH")
    assert [name for name, _ in events] == ["density_update"]
    assert services.worker.latest["RAMKUND_NORTH"]["risk_level"] == "YELLOW"


def test_local_source_maps_mediamtx_urls_to_clips(tmp_path):
    (tmp_path / "pandharpur_01.mp4").write_bytes(b"")
    (tmp_path / "ghat_cam.avi").write_bytes(b"")
    assert local_source("rtsp://localhost:8554/pandharpur_01", tmp_path) == str(tmp_path / "pandharpur_01.mp4")
    assert local_source("rtsp://localhost:8554/ghat_cam", tmp_path) == str(tmp_path / "ghat_cam.avi")
    assert local_source("rtsp://localhost:8554/missing", tmp_path) == "rtsp://localhost:8554/missing"
    assert local_source("rtsp://10.0.0.5/stream", tmp_path) == "rtsp://10.0.0.5/stream"


def test_local_frame_source_reads_clips_from_the_videos_folder(tmp_path, caplog):
    (tmp_path / "ramkund.mp4").write_bytes(b"")
    zones = [ZoneConfig("A", "A", "rtsp://localhost:8554/ramkund", area_m2=100.0),
             ZoneConfig("B", "B", "rtsp://localhost:8554/not_there", area_m2=100.0)]
    opened = []
    LocalFrameSource(zones, grabber_factory=lambda source, size: opened.append(source) or DummyGrabber(source),
                     videos=tmp_path)
    assert opened == [str(tmp_path / "ramkund.mp4"), "rtsp://localhost:8554/not_there"]
    assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == [
        f"zone B: no clip named not_there in {tmp_path}; trying the MediaMTX stream instead (needs Docker)"]


class DownSource(FakeSource):
    """Every zone fails like the Redis source does while Redis is down."""

    down = True

    def next_burst(self, zone_id):
        if self.down:
            raise FrameSourceUnavailable("Redis is not reachable at redis://localhost:6379/0 (Error 10061)")
        return super().next_burst(zone_id)


def test_an_unreachable_frame_source_is_logged_once_not_per_zone_per_cycle(caplog):
    caplog.set_level(logging.INFO)
    source = DownSource()
    services = make_services(source=source)
    for _ in range(3):
        assert services.worker.step() == []
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and warnings[0].startswith("Frame source unreachable: Redis is not reachable")
    assert services.worker.last_error.startswith("Frame source unreachable")

    source.down = False
    assert len(services.worker.step()) == 2  # both zones read again
    assert [r.getMessage() for r in caplog.records].count("frame source reachable again") == 1
    assert services.worker.last_error is None


class RefusingRedis:
    calls = 0

    def lpop(self, key):
        from redis.exceptions import ConnectionError as RedisConnectionError

        self.calls += 1
        raise RedisConnectionError("Error 10061 connecting to localhost:6379")


def test_redis_source_fails_fast_while_redis_is_down():
    client = RefusingRedis()
    source = RedisFrameSource(client, "redis://:secret@localhost:6379/0", retry_s=60)
    for _ in range(3):
        with pytest.raises(FrameSourceUnavailable, match="FRAME_SOURCE=local") as error:
            source.next_burst("A")
    assert client.calls == 1  # the next 60 s fail without waiting on the network
    assert "secret" not in str(error.value)
    source.retry_s, source._down = 0, None
    with pytest.raises(FrameSourceUnavailable):
        source.next_burst("A")
    assert client.calls == 2


def test_camera_outage_is_logged_once_while_retrying(tmp_path, caplog):
    grabber = FrameGrabber(str(tmp_path / "missing.mp4"), max_backoff_s=0.01)
    grabber.start()
    deadline = time.monotonic() + 5
    while grabber.reconnects < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    grabber.stop()
    grabber.join(timeout=5)
    assert grabber.reconnects >= 4 and grabber.status == "offline"
    assert len([r for r in caplog.records if "offline" in r.getMessage()]) == 1


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
