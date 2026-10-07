"""Camera ingestion (plan Layer 1): read each zone's stream, push 5-frame bursts to Redis.

    python -m app.camera_manager                  # cameras from config/zones.json
    python -m app.camera_manager --dummy --once   # synthetic frames: checks the Redis queue

One reader thread per camera decodes the stream continuously and keeps the newest frames
in a ring buffer. Every BURST_INTERVAL_S the manager pushes the latest BURST_SIZE
consecutive frames to the Redis list frame_queue:{zone_id}. The list is trimmed to the
newest bursts, so a slow inference worker gets fresh data instead of a growing backlog.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.config import ZoneConfig, get_settings, load_zones
from app.frame_codec import decode_burst, encode_burst

log = logging.getLogger(__name__)

QUEUE_KEY = "frame_queue:{zone_id}"
HEARTBEAT_KEY = "camera:{zone_id}:heartbeat"

Burst = tuple[list[np.ndarray], float, int]  # frames, capture time of the newest, frame number


def push_burst(redis_client, zone_id: str, blob: bytes, max_queue: int = 2) -> None:
    key = QUEUE_KEY.format(zone_id=zone_id)
    pipe = redis_client.pipeline()
    pipe.lpush(key, blob)
    pipe.ltrim(key, 0, max_queue - 1)
    pipe.execute()


def pop_burst(redis_client, zone_id: str, timeout: float | None = 1.0) -> bytes | None:
    """Newest burst first; older ones are dropped by the trim in push_burst.

    Waits up to `timeout` seconds for one; `timeout=None` returns at once.
    """
    key = QUEUE_KEY.format(zone_id=zone_id)
    if timeout is None:
        return redis_client.lpop(key)
    item = redis_client.blpop([key], timeout=timeout)
    return item[1] if item else None


def peek_burst(redis_client, zone_id: str) -> bytes | None:
    """The newest burst, left in the queue."""
    return redis_client.lindex(QUEUE_KEY.format(zone_id=zone_id), 0)


def write_heartbeat(redis_client, zone_id: str, info: dict, ttl_s: int = 90) -> None:
    redis_client.set(HEARTBEAT_KEY.format(zone_id=zone_id), json.dumps(info), ex=ttl_s)


class FrameGrabber(threading.Thread):
    """Continuously reads one source, keeping its newest `burst_size` frames.

    Reconnects with exponential backoff (1, 2, 4, 8 ... up to 30 s) when the stream drops.
    Network streams time out after `timeout_s` when opening or reading (OpenCV's default is
    30 s). Video files are paced at their own frame rate and looped, like a live camera.
    """

    def __init__(
        self,
        source: str,
        burst_size: int = 5,
        realtime: bool | None = None,
        max_backoff_s: float = 30.0,
        timeout_s: float = 5.0,
    ):
        super().__init__(daemon=True, name=f"grabber:{source}")
        self.source = source
        self.burst_size = burst_size
        self.is_file = "://" not in source
        self.realtime = self.is_file if realtime is None else realtime
        self.max_backoff_s = max_backoff_s
        self.timeout_s = timeout_s
        self.status = "connecting"
        self.frames_read = 0
        self.reconnects = 0
        self.last_frame_time: float | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=burst_size)
        self._lock = threading.Lock()
        self._stopping = threading.Event()

    def stop(self) -> None:
        self._stopping.set()

    def latest_burst(self) -> Burst | None:
        with self._lock:
            if len(self._frames) < self.burst_size:
                return None
            return list(self._frames), self.last_frame_time, self.frames_read

    def _open(self) -> cv2.VideoCapture:
        if self.is_file:
            return cv2.VideoCapture(self.source)
        timeout_ms = int(self.timeout_s * 1000)
        return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
        ])

    def run(self) -> None:
        backoff = 1.0
        while not self._stopping.is_set():
            cap = self._open()
            if cap.isOpened():
                self.status = "online"
                backoff = 1.0
                self._read_until_failure(cap)
            cap.release()
            if self._stopping.is_set():
                break
            self.status = "offline"
            self.reconnects += 1
            log.warning("camera %s offline; retrying in %.0f s", self.source, backoff)
            self._stopping.wait(backoff)
            backoff = min(backoff * 2, self.max_backoff_s)

    def _read_until_failure(self, cap: cv2.VideoCapture) -> None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        period = 1.0 / (fps if 1 <= fps <= 120 else 25.0)
        next_time = time.monotonic()
        while not self._stopping.is_set():
            ok, frame = cap.read()
            if not ok and self.is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                with self._lock:
                    self._frames.clear()  # never mix a clip's end and start in one burst
                ok, frame = cap.read()
            if not ok:
                return
            with self._lock:
                self._frames.append(frame)
                self.frames_read += 1
                self.last_frame_time = time.time()
            if self.realtime:
                next_time += period
                delay = next_time - time.monotonic()
                if delay > 0:
                    self._stopping.wait(delay)
                else:
                    next_time = time.monotonic()  # fell behind: don't try to catch up


class DummyGrabber:
    """Synthetic camera (grey frames with drifting dots) for testing without video."""

    def __init__(self, source: str, burst_size: int = 5, size: tuple[int, int] = (640, 360)):
        self.source = source
        self.burst_size = burst_size
        self.size = size
        self.status = "online"
        self.frames_read = 0
        self.reconnects = 0
        self.last_frame_time: float | None = None
        self._dots = np.random.default_rng(0).uniform((0, 0), size, (200, 2))

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def latest_burst(self) -> Burst:
        frames = []
        for _ in range(self.burst_size):
            self._dots[:, 0] = (self._dots[:, 0] + 2) % self.size[0]
            frame = np.full((self.size[1], self.size[0], 3), 90, np.uint8)
            for x, y in self._dots.astype(int):
                cv2.circle(frame, (int(x), int(y)), 3, (30, 30, 30), -1)
            frames.append(frame)
        self.frames_read += self.burst_size
        self.last_frame_time = time.time()
        return frames, self.last_frame_time, self.frames_read


def _limit_width(frame: np.ndarray, max_width: int | None) -> np.ndarray:
    h, w = frame.shape[:2]
    if not max_width or w <= max_width:
        return frame
    return cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)


class CameraManager:
    def __init__(
        self,
        zones: list[ZoneConfig],
        redis_client,
        burst_size: int = 5,
        interval_s: float = 2.0,
        max_width: int | None = 1280,
        jpeg_quality: int = 90,
        heartbeat_s: float = 30.0,
        grabber_factory: Callable[..., object] = FrameGrabber,
    ) -> None:
        self.redis = redis_client
        self.grabbers = {zone.id: grabber_factory(zone.source, burst_size) for zone in zones}
        self.interval_s = interval_s
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality
        self.heartbeat_s = heartbeat_s
        self._last_pushed: dict[str, int] = {}

    def start(self) -> None:
        for grabber in self.grabbers.values():
            grabber.start()

    def stop(self) -> None:
        for grabber in self.grabbers.values():
            grabber.stop()

    def tick(self) -> dict[str, bool]:
        """Push one fresh burst per camera; returns which zones pushed."""
        pushed = {}
        for zone_id, grabber in self.grabbers.items():
            burst = grabber.latest_burst()
            if burst is None or burst[2] == self._last_pushed.get(zone_id):
                pushed[zone_id] = False  # not enough frames yet, or the stream has stalled
                continue
            frames, captured_at, frame_no = burst
            meta = {"zone_id": zone_id, "captured_at": captured_at, "frame_no": frame_no}
            frames = [_limit_width(f, self.max_width) for f in frames]
            push_burst(self.redis, zone_id, encode_burst(frames, meta, self.jpeg_quality))
            self._last_pushed[zone_id] = frame_no
            pushed[zone_id] = True
        return pushed

    def heartbeat(self) -> None:
        for zone_id, g in self.grabbers.items():
            write_heartbeat(self.redis, zone_id, {
                "status": g.status, "last_frame_time": g.last_frame_time, "frames_read": g.frames_read,
                "reconnects": g.reconnects, "updated": time.time(),
            })

    def run_forever(self) -> None:
        self.start()
        next_tick = next_heartbeat = time.monotonic()
        try:
            while True:
                if time.monotonic() >= next_heartbeat:
                    self.heartbeat()
                    next_heartbeat += self.heartbeat_s
                self.tick()
                next_tick += self.interval_s
                time.sleep(max(0.0, next_tick - time.monotonic()))
        finally:
            self.stop()


def main(argv=None) -> None:
    import redis

    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zones", type=Path, default=settings.zones_config)
    p.add_argument("--dummy", action="store_true", help="synthetic frames instead of real cameras")
    p.add_argument("--once", action="store_true", help="push one burst per zone, read it back, exit")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = redis.Redis.from_url(settings.redis_url)
    try:
        client.ping()
    except redis.ConnectionError:
        raise SystemExit(f"Redis is not reachable at {settings.redis_url}; start it with: docker compose up -d redis")

    manager = CameraManager(
        load_zones(args.zones), client, settings.burst_size, settings.burst_interval_s, settings.infer_max_width,
        grabber_factory=DummyGrabber if args.dummy else FrameGrabber,
    )
    if not args.once:
        manager.run_forever()
        return

    manager.start()
    deadline = time.monotonic() + 15
    pending = set(manager.grabbers)
    while pending and time.monotonic() < deadline:
        pending -= {zone for zone, ok in manager.tick().items() if ok}
        time.sleep(0.2)
    manager.heartbeat()
    manager.stop()
    for zone_id in manager.grabbers:
        blob = pop_burst(client, zone_id, timeout=1)
        if blob is None:
            print(f"{zone_id}: no burst (is the camera/stream reachable?)")
            continue
        frames, meta = decode_burst(blob)
        h, w = frames[0].shape[:2]
        print(f"{zone_id}: {len(frames)} frames {w}x{h}, frame_no {meta['frame_no']}, {len(blob) / 1024:.0f} KiB - OK")


if __name__ == "__main__":
    main()
