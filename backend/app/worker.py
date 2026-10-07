"""Inference worker (plan Layers 2-3): frame bursts in; risk, exits, alerts and events out.

main.py calls step() in a thread every BURST_INTERVAL_S (2 s). For each zone with a new burst:
CSRNet on the middle frame + optical flow across the burst -> RiskEngine -> exit-router
pressure -> stored reading -> alert decision (logged to AlertLog). step() returns the
Socket.IO events ("density_update", "alert") for main.py to emit.

Frames come from Redis (pushed by `python -m app.camera_manager`) or, with FRAME_SOURCE=local,
straight from the cameras or video files inside this process, which needs no Docker.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from app.alert_dispatcher import AlertDispatcher, AlertPolicy, build_channels
from app.camera_manager import HEARTBEAT_KEY, FrameGrabber, peek_burst, pop_burst
from app.config import DATA_DIR, Settings, ZoneConfig, load_zones
from app.exit_router import ExitRouter
from app.frame_codec import decode_burst
from app.risk_engine import ZoneStatus
from app.store import Store, open_store

log = logging.getLogger(__name__)

Burst = tuple[list[np.ndarray], float]  # frames, capture time (epoch seconds)
Event = tuple[str, dict]  # Socket.IO event name, payload

RAW_VIDEOS = DATA_DIR / "india_dataset" / "raw_videos"
MEDIAMTX_PREFIXES = ("rtsp://localhost:8554/", "rtsp://127.0.0.1:8554/")


class FrameSource(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def next_burst(self, zone_id: str) -> Burst | None: ...  # newest burst not yet processed
    def peek_burst(self, zone_id: str) -> Burst | None: ...  # newest burst, processed or not
    def camera_status(self, zone_id: str) -> dict | None: ...


class RedisFrameSource:
    """Bursts pushed to Redis by the camera manager."""

    def __init__(self, redis_client) -> None:
        self.redis = redis_client

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def next_burst(self, zone_id: str) -> Burst | None:
        return self._decode(pop_burst(self.redis, zone_id, timeout=None))

    def peek_burst(self, zone_id: str) -> Burst | None:
        return self._decode(peek_burst(self.redis, zone_id))

    def camera_status(self, zone_id: str) -> dict | None:
        raw = self.redis.get(HEARTBEAT_KEY.format(zone_id=zone_id))
        return json.loads(raw) if raw else None

    @staticmethod
    def _decode(blob: bytes | None) -> Burst | None:
        if blob is None:
            return None
        frames, meta = decode_burst(blob)
        return frames, float(meta.get("captured_at") or time.time())


def local_source(source: str, videos: Path = RAW_VIDEOS) -> str:
    """For rtsp://localhost:8554/<name>: the clip MediaMTX would replay, if it exists here."""
    for prefix in MEDIAMTX_PREFIXES:
        if source.startswith(prefix):
            clip = videos / f"{source[len(prefix):]}.mp4"
            if clip.exists():
                return str(clip)
    return source


class LocalFrameSource:
    """Reads every camera or video file in this process: no Redis or MediaMTX needed."""

    def __init__(self, zones: Sequence[ZoneConfig], burst_size: int = 5,
                 grabber_factory: Callable[..., FrameGrabber] = FrameGrabber) -> None:
        self.grabbers = {z.id: grabber_factory(local_source(z.source), burst_size) for z in zones}
        self._last: dict[str, int] = {}

    def start(self) -> None:
        for grabber in self.grabbers.values():
            grabber.start()

    def stop(self) -> None:
        for grabber in self.grabbers.values():
            grabber.stop()

    def next_burst(self, zone_id: str) -> Burst | None:
        burst = self.grabbers[zone_id].latest_burst()
        if burst is None or burst[2] == self._last.get(zone_id):
            return None  # not enough frames yet, or nothing new since the last step
        self._last[zone_id] = burst[2]
        return burst[0], burst[1] or time.time()

    def peek_burst(self, zone_id: str) -> Burst | None:
        burst = self.grabbers[zone_id].latest_burst()
        return (burst[0], burst[1] or time.time()) if burst else None

    def camera_status(self, zone_id: str) -> dict | None:
        g = self.grabbers[zone_id]
        return {"status": g.status, "last_frame_time": g.last_frame_time, "frames_read": g.frames_read,
                "reconnects": g.reconnects}


def graph_node(zone: ZoneConfig, router: ExitRouter) -> str | None:
    """The zone's node in the exit graph: its graph_node, else its id if that is a node."""
    node = zone.graph_node or zone.id
    return node if node in router.graph else None


def density_payload(zone: ZoneConfig, status: ZoneStatus, exits: list[dict]) -> dict:
    """Body of the "density_update" event and of GET /api/zones/{id}/status."""
    return {
        "zone_id": zone.id, "name": zone.name, "timestamp": status.timestamp,
        "density": status.density, "turbulence": status.turbulence,
        "pressure": status.smoothed_pressure, "raw_pressure": status.pressure,
        "risk_level": str(status.level), "count": status.count, "samples": status.samples, "exits": exits,
    }


class InferenceWorker:
    def __init__(self, zones: Sequence[ZoneConfig], analyzers: dict, source: FrameSource, router: ExitRouter,
                 dispatcher: AlertDispatcher, store: Store) -> None:
        self.zones = {z.id: z for z in zones}
        self.analyzers = analyzers  # zone id -> app.pipeline.ZoneAnalyzer
        self.source = source
        self.router = router
        self.dispatcher = dispatcher
        self.store = store
        self.latest: dict[str, dict] = {}  # zone id -> last density payload
        self.last_error: str | None = None
        self._lock = threading.Lock()

    def step(self) -> list[Event]:
        """Process the newest burst of every zone; returns the events to emit."""
        events: list[Event] = []
        with self._lock:
            for zone in self.zones.values():
                try:
                    burst = self.source.next_burst(zone.id)
                except Exception as exc:  # e.g. Redis down: keep serving the other zones
                    self._fail(f"{zone.id}: cannot read frames ({exc})")
                    continue
                if burst is not None:
                    events += self._process(zone, *burst)
        return events

    def infer_now(self, zone_id: str) -> list[Event] | None:
        """Run one zone on its newest burst immediately; None if it has no frames yet."""
        burst = self.source.peek_burst(zone_id)
        if burst is None:
            return None
        with self._lock:
            return self._process(self.zones[zone_id], *burst)

    def _process(self, zone: ZoneConfig, frames: list[np.ndarray], captured_at: float) -> list[Event]:
        try:
            status = self.analyzers[zone.id].analyze(frames, captured_at).status
        except Exception as exc:
            log.exception("inference failed for zone %s", zone.id)
            self.last_error = f"{zone.id}: inference failed ({exc})"
            return []
        node = graph_node(zone, self.router)
        if node:
            self.router.update_pressure(node, status.smoothed_pressure)
        exits = self.router.get_best_exits(node) if node else []
        payload = density_payload(zone, status, exits)
        self.latest[zone.id] = payload
        self._try(self.store.record_reading, status)
        events: list[Event] = [("density_update", payload)]

        alert = self.dispatcher.handle(status, zone.name, exits[0]["path"] if exits else None)
        if alert is not None:
            self._try(self.store.record_alert, alert)
            events.append(("alert", {
                "zone_id": zone.id, "risk_level": str(alert.level), "message": alert.text,
                "exits": [e["exit"] for e in exits], "timestamp": alert.triggered_at,
                "delivered": any(d.ok for d in alert.deliveries),
            }))
        return events

    def _try(self, write: Callable, item) -> None:
        try:
            write(item)
        except Exception as exc:  # a database error must never stop the inference loop
            self._fail(f"storage: {exc}")

    def _fail(self, message: str) -> None:
        log.warning(message)
        self.last_error = message


@dataclass
class Services:
    settings: Settings
    zones: dict[str, ZoneConfig]
    router: ExitRouter
    store: Store
    source: FrameSource
    worker: InferenceWorker | None  # None while the CSRNet model is unavailable
    model_error: str | None = None


def make_frame_source(settings: Settings, zones: Sequence[ZoneConfig]) -> FrameSource:
    if settings.frame_source == "local":
        return LocalFrameSource(zones, settings.burst_size)
    if settings.frame_source == "redis":
        import redis

        return RedisFrameSource(redis.Redis.from_url(settings.redis_url, socket_timeout=2))
    raise ValueError(f"FRAME_SOURCE must be 'redis' or 'local', not {settings.frame_source!r}")


def build_services(settings: Settings) -> Services:
    """Everything main.py needs. A missing or broken model only disables inference."""
    zones = load_zones(settings.zones_config)
    router = ExitRouter()
    for z in zones:
        if z.graph_node and z.graph_node not in router.graph:
            raise ValueError(f"zone {z.id!r}: graph_node {z.graph_node!r} is not in the exit graph")
    store = open_store(settings.database_url)
    try:
        store.sync_zones(zones)
    except Exception as exc:
        log.warning("could not save zones to the database: %s", exc)
    source = make_frame_source(settings, zones)
    by_id = {z.id: z for z in zones}

    if not settings.csrnet_weights.exists():
        error = f"CSRNet weights not found at {settings.csrnet_weights}; train a model or set CSRNET_WEIGHTS in .env"
        log.warning(error)
        return Services(settings, by_id, router, store, source, None, error)
    try:
        from app.csrnet_model import DensityEstimator
        from app.pipeline import ZoneAnalyzer

        estimator = DensityEstimator(settings.csrnet_weights, settings.device, settings.infer_max_width,
                                     allow_pickle=settings.csrnet_allow_pickle)
        analyzers = {z.id: ZoneAnalyzer(z, estimator, settings) for z in zones}
    except Exception as exc:  # corrupt file, old pickle without CSRNET_ALLOW_PICKLE, ...
        error = f"could not load CSRNet weights {settings.csrnet_weights}: {exc}"
        log.warning(error)
        return Services(settings, by_id, router, store, source, None, error)

    dispatcher = AlertDispatcher(build_channels(settings), AlertPolicy(settings.alert_cooldown_s))
    worker = InferenceWorker(zones, analyzers, source, router, dispatcher, store)
    return Services(settings, by_id, router, store, source, worker)
