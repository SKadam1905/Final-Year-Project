import json
import time

import fakeredis
import numpy as np
import pytest

from app.camera_manager import (
    HEARTBEAT_KEY,
    QUEUE_KEY,
    CameraManager,
    DummyGrabber,
    FrameGrabber,
    peek_burst,
    pop_burst,
    push_burst,
    write_heartbeat,
)
from app.config import ZoneConfig
from app.frame_codec import decode_burst, encode_burst
from tests.conftest import write_video


def gradient_frames(n=5):
    ramp = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (48, 1))
    return [np.dstack([np.roll(ramp, k, axis=1)] * 3) for k in range(n)]


def test_codec_roundtrip():
    frames = gradient_frames()
    frames_out, meta = decode_burst(encode_burst(frames, {"zone_id": "z1", "frame_no": 7}, quality=95))
    assert meta == {"zone_id": "z1", "frame_no": 7}
    assert len(frames_out) == 5
    for a, b in zip(frames, frames_out):
        assert a.shape == b.shape
        assert np.abs(a.astype(int) - b.astype(int)).mean() < 3


def test_codec_rejects_garbage():
    blob = encode_burst(gradient_frames(2), {})
    with pytest.raises(ValueError):
        decode_burst(b"nope" + blob[4:])
    with pytest.raises(ValueError):
        decode_burst(blob[:-100])


def test_queue_keeps_only_newest_bursts():
    r = fakeredis.FakeRedis()
    for i in range(3):
        push_burst(r, "z1", f"burst{i}".encode(), max_queue=2)
    assert r.llen(QUEUE_KEY.format(zone_id="z1")) == 2
    assert peek_burst(r, "z1") == b"burst2"  # peeking leaves it queued
    assert pop_burst(r, "z1") == b"burst2"
    assert pop_burst(r, "z1", timeout=None) == b"burst1"
    assert pop_burst(r, "z1", timeout=None) is None  # returns at once when empty
    assert pop_burst(r, "z1", timeout=0.1) is None


def test_heartbeat_expires():
    r = fakeredis.FakeRedis()
    write_heartbeat(r, "z1", {"status": "online"}, ttl_s=90)
    key = HEARTBEAT_KEY.format(zone_id="z1")
    assert json.loads(r.get(key)) == {"status": "online"}
    assert 0 < r.ttl(key) <= 90


def test_frame_grabber_returns_consecutive_frames(tmp_path):
    # Frame k is uniformly grey with value 5k, so consecutive frames differ by 5.
    frames = [np.full((48, 64, 3), 5 * k, np.uint8) for k in range(40)]
    video = write_video(tmp_path / "clip.avi", frames)
    grabber = FrameGrabber(str(video), burst_size=5, realtime=False)
    grabber.start()
    try:
        deadline = time.monotonic() + 10
        while (burst := grabber.latest_burst()) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert burst is not None, "grabber produced no burst"
        values = [int(round(f.mean())) for f in burst[0]]
        assert all(abs((b - a) - 5) <= 2 for a, b in zip(values, values[1:])), values
        assert grabber.status == "online"
    finally:
        grabber.stop()
        grabber.join(timeout=5)


def test_grabber_reports_offline_source():
    grabber = FrameGrabber("rtsp://127.0.0.1:1/none", max_backoff_s=0.1, timeout_s=2.0)
    started = time.monotonic()
    grabber.start()
    try:
        while grabber.reconnects == 0 and time.monotonic() - started < 15:
            time.sleep(0.05)
        assert grabber.reconnects >= 1, "open did not time out"
        assert time.monotonic() - started < 10  # not OpenCV's default 30 s
        assert grabber.status in ("offline", "connecting")  # it may already be retrying
        assert grabber.latest_burst() is None
    finally:
        grabber.stop()
        grabber.join(timeout=10)


def zones(n=2):
    return [ZoneConfig(id=f"z{i}", name=f"Zone {i}", source=f"dummy{i}", area_m2=100.0) for i in range(1, n + 1)]


def test_manager_pushes_decodable_bursts():
    r = fakeredis.FakeRedis()
    manager = CameraManager(zones(), r, burst_size=5, max_width=320, grabber_factory=DummyGrabber)
    assert manager.tick() == {"z1": True, "z2": True}
    manager.heartbeat()
    frames, meta = decode_burst(pop_burst(r, "z2"))
    assert len(frames) == 5 and frames[0].shape == (180, 320, 3) and meta["zone_id"] == "z2"
    assert json.loads(r.get(HEARTBEAT_KEY.format(zone_id="z1")))["status"] == "online"


class StalledGrabber(DummyGrabber):
    def latest_burst(self):
        frames, _, _ = super().latest_burst()
        return frames, 0.0, 42  # the frame number never advances


def test_manager_skips_stalled_streams():
    manager = CameraManager(zones(1), fakeredis.FakeRedis(), grabber_factory=StalledGrabber)
    assert manager.tick() == {"z1": True}
    assert manager.tick() == {"z1": False}
