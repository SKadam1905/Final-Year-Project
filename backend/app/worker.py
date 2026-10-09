"""Inference worker (plan Layers 2-3): frame bursts in; risk, exits, alerts and events out.

main.py calls step() in a thread every BURST_INTERVAL_S (2 s). For each zone with a new burst:
CSRNet on the middle frame + optical flow across the burst -> RiskEngine -> exit-router
pressure -> stored reading -> alert decision (logged to AlertLog). step() returns the
Socket.IO events ("density_update", "alert") for main.py to emit.

Frames come straight from the cameras or video files inside this process (FRAME_SOURCE=local, the
default: no Docker needed) or from Redis (FRAME_SOURCE=redis, pushed by `python -m app.camera_manager`).
An unreachable frame source is logged once, not once per zone per cycle, until it is back.
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
from app.config import DATA_DIR, Settings, ZoneConfig, load_zones, redact
from app.exit_router import ExitRouter
from app.gate_actions import gate_actions
from app.frame_codec import decode_burst
from app.risk_engine import ZoneStatus
from app.store import Store, alert_dict, open_store

log = logging.getLogger(__name__)

Burst = tuple[list[np.ndarray], float]  # frames, capture time (epoch seconds)
Event = tuple[str, dict]  # Socket.IO event name, payload

RAW_VIDEOS = DATA_DIR / "india_dataset" / "raw_videos"
MEDIAMTX_PREFIXES = ("rtsp://localhost:8554/", "rtsp://127.0.0.1:8554/")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


class FrameSourceUnavailable(RuntimeError):
    """The frame source cannot be reached; the message says what to do about it."""


class FrameSource(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def next_burst(self, zone_id: str) -> Burst | None: ...  # newest burst not yet processed
    def peek_burst(self, zone_id: str) -> Burst | None: ...  # newest burst, processed or not
    def camera_status(self, zone_id: str) -> dict | None: ...


class RedisFrameSource:
    """Bursts pushed to Redis by the camera manager.

    After a connection error every call fails at once with FrameSourceUnavailable for `retry_s`
    seconds: a refused connection takes ~2 s on Windows, and every zone asks every cycle.
    """

    def __init__(self, redis_client, url: str = "", retry_s: float = 5.0) -> None:
        from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeout

        self.redis = redis_client
        self.url = url
        self.retry_s = retry_s
        self._errors = (RedisConnectionError, RedisTimeout, OSError)
        self._down: tuple[float, str] | None = None  # (retry after, reason) while unreachable

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def _call(self, fn: Callable, *args):
        if self._down and time.monotonic() < self._down[0]:
            raise FrameSourceUnavailable(self._down[1])
        try:
            result = fn(*args)
        except self._errors as exc:
            reason = (f"Redis is not reachable at {redact(self.url)} ({exc}). Start Redis and the camera manager, "
                      "or set FRAME_SOURCE=local in .env to read the cameras and video files directly (no Docker)")
            self._down = (time.monotonic() + self.retry_s, reason)
            raise FrameSourceUnavailable(reason) from exc
        self._down = None
        return result

    def next_burst(self, zone_id: str) -> Burst | None:
        return self._decode(self._call(pop_burst, self.redis, zone_id, None))

    def peek_burst(self, zone_id: str) -> Burst | None:
        return self._decode(self._call(peek_burst, self.redis, zone_id))

    def camera_status(self, zone_id: str) -> dict | None:
        raw = self._call(self.redis.get, HEARTBEAT_KEY.format(zone_id=zone_id))
        return json.loads(raw) if raw else None

    @staticmethod
    def _decode(blob: bytes | None) -> Burst | None:
        if blob is None:
            return None
        frames, meta = decode_burst(blob)
        return frames, float(meta.get("captured_at") or time.time())


def local_source(source: str, videos: Path = RAW_VIDEOS) -> str:
    """For rtsp://localhost:8554/<name>: the clip MediaMTX would replay (<name>.mp4, or .avi/.mov/.mkv,
    in data/india_dataset/raw_videos), if it is there; any other source unchanged."""
    for prefix in MEDIAMTX_PREFIXES:
        if source.startswith(prefix):
            for ext in VIDEO_EXTENSIONS:
                clip = videos / f"{source[len(prefix):]}{ext}"
                if clip.exists():
                    return str(clip)
    return source


class LocalFrameSource:
    """Reads every camera or video file in this process: no Redis or MediaMTX needed."""

    def __init__(self, zones: Sequence[ZoneConfig], burst_size: int = 5,
                 grabber_factory: Callable[..., FrameGrabber] = FrameGrabber, videos: Path = RAW_VIDEOS) -> None:
        self.grabbers = {}
        for z in zones:
            source = local_source(z.source, videos)
            if source.startswith(MEDIAMTX_PREFIXES):
                log.warning("zone %s: no clip named %s in %s; trying the MediaMTX stream instead (needs Docker)",
                            z.id, source.rsplit("/", 1)[-1], videos)
            elif "://" not in source and not Path(source).exists():
                log.warning("zone %s: video file %s not found", z.id, source)
            self.grabbers[z.id] = grabber_factory(source, burst_size)
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


class WithUploadedVideos:
    """The cameras' frame source, plus uploaded videos looping as extra zones.

    Uploaded videos are always read in this process (whatever FRAME_SOURCE is), paced at their
    own frame rate like a camera; every other zone goes to the cameras' source unchanged.
    """

    def __init__(self, cameras: FrameSource, burst_size: int = 5,
                 grabber_factory: Callable[..., FrameGrabber] = FrameGrabber) -> None:
        self.cameras = cameras
        self.videos = LocalFrameSource([], burst_size, grabber_factory)
        self._grabber_factory = grabber_factory
        self._burst_size = burst_size

    def add_video(self, zone: ZoneConfig) -> None:
        grabber = self._grabber_factory(zone.source, self._burst_size)
        self.videos.grabbers[zone.id] = grabber
        grabber.start()

    def remove_video(self, zone_id: str) -> None:
        """Stop reading the video; returns once its reader has closed the file (so it can be deleted)."""
        grabber = self.videos.grabbers.pop(zone_id, None)
        self.videos._last.pop(zone_id, None)
        if grabber is not None:
            grabber.stop()
            if isinstance(grabber, threading.Thread) and grabber.is_alive():
                grabber.join(timeout=5)

    def _source(self, zone_id: str) -> FrameSource:
        return self.videos if zone_id in self.videos.grabbers else self.cameras

    def start(self) -> None:
        self.cameras.start()

    def stop(self) -> None:
        self.cameras.stop()
        self.videos.stop()

    def next_burst(self, zone_id: str) -> Burst | None:
        return self._source(zone_id).next_burst(zone_id)

    def peek_burst(self, zone_id: str) -> Burst | None:
        return self._source(zone_id).peek_burst(zone_id)

    def camera_status(self, zone_id: str) -> dict | None:
        return self._source(zone_id).camera_status(zone_id)


def graph_node(zone: ZoneConfig, router: ExitRouter) -> str | None:
    """The zone's node in the exit graph: its graph_node, else its id if that is a node."""
    node = zone.graph_node or zone.id
    return node if node in router.graph else None


def density_payload(zone: ZoneConfig, status: ZoneStatus, exits: list[dict], grid: dict | None = None,
                    actions: list[dict] | None = None) -> dict:
    """Body of the "density_update" event and of GET /api/zones/{id}/status."""
    return {
        "zone_id": zone.id, "name": zone.name, "timestamp": status.timestamp,
        "density": status.density, "turbulence": status.turbulence,
        "pressure": status.smoothed_pressure, "raw_pressure": status.pressure,
        "risk_level": str(status.level), "count": status.count, "samples": status.samples, "exits": exits,
        "count_method": status.count_method, "density_count": status.density_count,
        "detection_count": status.detection_count, "grid": grid, "hotspot": status.hotspot,
        "gate_actions": actions or [],
    }


@dataclass(frozen=True)
class ZoneFrame:
    """The frame CSRNet last analysed for a zone, for the snapshot and overlay endpoints."""

    frame: np.ndarray  # BGR, the middle frame of the burst
    density: np.ndarray | None  # CSRNet's density map of it (None if not analysed)
    timestamp: float
    boxes: np.ndarray | None = None  # YOLO person boxes (N, 4) in frame pixels (None without a detector)
    grid: dict | None = None  # per-cell counts (ZoneGrid.to_dict)


@dataclass
class ZoneOverride:
    """A zone added at runtime (an uploaded test video) gets its own alert dispatcher and its own
    copy of the exit graph, so it never changes the exit advice given for the real zones."""

    dispatcher: AlertDispatcher
    router: ExitRouter
    node: str | None  # its position on that copy of the graph


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
        self.frames: dict[str, ZoneFrame] = {}  # zone id -> last analysed frame
        self.last_error: str | None = None
        # One lock per zone: zones can run in parallel threads (main.py does), but one zone's
        # risk engine and alert state are never updated by two threads at once.
        self._locks = {z.id: threading.Lock() for z in zones}
        self.overrides: dict[str, ZoneOverride] = {}
        self._unreachable: dict[str, str] = {}  # zone id -> why its frames cannot be read (logged once)
        self._unreachable_lock = threading.Lock()

    def add_zone(self, zone: ZoneConfig, analyzer, override: ZoneOverride) -> None:
        """Start analysing a zone added at runtime (its frames must already be in the source)."""
        self._locks[zone.id] = threading.Lock()
        self.analyzers[zone.id] = analyzer
        self.overrides[zone.id] = override
        self.zones[zone.id] = zone

    def remove_zone(self, zone_id: str) -> None:
        """Stop analysing a runtime zone; waits for a step already running on it."""
        lock = self._locks.get(zone_id)
        if lock is None:
            return
        with lock:
            for table in (self.zones, self.analyzers, self.overrides, self.latest, self.frames):
                table.pop(zone_id, None)
            del self._locks[zone_id]
        with self._unreachable_lock:
            self._unreachable.pop(zone_id, None)

    def step(self) -> list[Event]:
        """Process the newest burst of every zone, one after another; returns the events."""
        return [event for zone_id in list(self.zones) for event in self.step_zone(zone_id)]

    def step_zone(self, zone_id: str) -> list[Event]:
        """Process the newest burst of one zone, if a new one has arrived."""
        lock = self._locks.get(zone_id)
        if lock is None:
            return []  # removed while this cycle was being scheduled
        with lock:
            zone = self.zones.get(zone_id)
            if zone is None:
                return []
            try:
                burst = self.source.next_burst(zone_id)
            except Exception as exc:  # e.g. Redis down: keep serving the other zones
                self._source_failed(zone_id, exc)
                return []
            self._source_ok(zone_id)
            return self._process(zone, *burst) if burst is not None else []

    def _source_failed(self, zone_id: str, exc: Exception) -> None:
        """Log a frame-source failure once for all zones it affects, not once per zone per cycle."""
        message = f"Frame source unreachable: {exc}"
        with self._unreachable_lock:
            first = message not in self._unreachable.values()
            self._unreachable[zone_id] = message
        self.last_error = message
        if first:
            log.warning("%s. Live zones get no new readings until it is back (logged once).", message)

    def _source_ok(self, zone_id: str) -> None:
        with self._unreachable_lock:
            if self._unreachable.pop(zone_id, None) is None or self._unreachable:
                return
        log.info("frame source reachable again")
        if self.last_error and self.last_error.startswith("Frame source unreachable"):
            self.last_error = None

    def infer_now(self, zone_id: str) -> list[Event] | None:
        """Run one zone on its newest burst immediately; None if it has no frames yet."""
        burst = self.source.peek_burst(zone_id)
        if burst is None:
            return None
        with self._locks[zone_id]:
            return self._process(self.zones[zone_id], *burst)

    def _process(self, zone: ZoneConfig, frames: list[np.ndarray], captured_at: float) -> list[Event]:
        try:
            analysis = self.analyzers[zone.id].analyze(frames, captured_at)
        except Exception as exc:
            log.exception("inference failed for zone %s", zone.id)
            self.last_error = f"{zone.id}: inference failed ({exc})"
            return []
        if analysis.scene_cut:  # the camera switched shots mid-burst: no reading this cycle
            log.info("zone %s: scene cut in the burst, skipped", zone.id)
            return []
        status = analysis.status
        grid = analysis.grid.to_dict() if analysis.grid is not None else None
        boxes = analysis.detections.boxes if analysis.detections is not None else None
        self.frames[zone.id] = ZoneFrame(frames[len(frames) // 2], analysis.density.density, captured_at, boxes, grid)
        override = self.overrides.get(zone.id)
        router = override.router if override else self.router
        node = override.node if override else graph_node(zone, self.router)
        if node:
            router.update_pressure(node, status.smoothed_pressure)
        exits = router.get_best_exits(node) if node else []
        actions = gate_actions(status.level, router, node, exits, status.hotspot)
        payload = density_payload(zone, status, exits, grid, actions)
        self.latest[zone.id] = payload
        self._try(self.store.record_reading, status)
        events: list[Event] = [("density_update", payload)]

        dispatcher = override.dispatcher if override else self.dispatcher
        alert = dispatcher.handle(status, zone.name, exits[0]["path"] if exits else None,
                                  [a["text"] for a in actions])
        if alert is not None:
            stored = self._try(self.store.record_alert, alert)
            events.append(("alert", stored or alert_dict(None, alert, None)))
        return events

    def _try(self, write: Callable, item):
        try:
            return write(item)
        except Exception as exc:  # a database error must never stop the inference loop
            self._fail(f"storage: {exc}")
            return None

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
    gradcam: object | None = None  # ml.gradcam.GradCAM over the loaded model
    estimator: object | None = None  # the DensityEstimator the worker uses (video analysis shares it)
    detector: object | None = None  # app.detector.PersonDetector (hybrid count), shared the same way
    detector_error: str | None = None  # why there is no detector


def make_frame_source(settings: Settings, zones: Sequence[ZoneConfig]) -> FrameSource:
    if settings.frame_source == "local":
        log.info("frames: read in this process (FRAME_SOURCE=local); clips from %s", RAW_VIDEOS)
        return LocalFrameSource(zones, settings.burst_size)
    if settings.frame_source == "redis":
        import redis

        log.info("frames: from Redis at %s (FRAME_SOURCE=redis; needs Docker)", redact(settings.redis_url))
        client = redis.Redis.from_url(settings.redis_url, socket_timeout=2, socket_connect_timeout=2)
        return RedisFrameSource(client, settings.redis_url)
    raise ValueError(f"FRAME_SOURCE must be 'local' or 'redis', not {settings.frame_source!r}")


def build_services(settings: Settings) -> Services:
    """Everything main.py needs. A missing or broken model only disables inference."""
    zones = load_zones(settings.zones_config)
    router = ExitRouter()
    for z in zones:
        if z.graph_node and z.graph_node not in router.graph:
            raise ValueError(f"zone {z.id!r}: graph_node {z.graph_node!r} is not in the exit graph")
    store = open_store(settings.mongodb_uri, settings.mongodb_db, show_demo=settings.demo_mode)
    try:
        store.sync_zones(zones)
    except Exception as exc:
        log.warning("could not save zones to the database: %s", exc)
    source = WithUploadedVideos(make_frame_source(settings, zones), settings.burst_size)
    by_id = {z.id: z for z in zones}

    if not settings.csrnet_weights.exists():
        error = (f"No trained model: counts unavailable. CSRNet weights not found at {settings.csrnet_weights}; "
                 "train a model or set CSRNET_WEIGHTS in .env")
        log.warning(error)
        return Services(settings, by_id, router, store, source, None, error)
    try:
        from app.csrnet_model import DensityEstimator
        from app.pipeline import ZoneAnalyzer
        from ml.gradcam import GradCAM

        from app.csrnet_model import TRUSTED_KINDS

        estimator = DensityEstimator(settings.csrnet_weights, settings.device, settings.infer_max_width,
                                     allow_pickle=settings.csrnet_allow_pickle)
        if estimator.info["kind"] not in TRUSTED_KINDS and not settings.demo_mode:
            error = (f"No trained model: counts unavailable. {settings.csrnet_weights.name} is "
                     f"{estimator.info['kind']} weights ({estimator.info['label']}) and is refused; set CSRNET_WEIGHTS "
                     "to trained or imported public weights, or DEMO_MODE=1 in .env for a demo.")
            log.warning(error)
            return Services(settings, by_id, router, store, source, None, error)
        gradcam = GradCAM(estimator.model)
    except Exception as exc:  # corrupt file, old pickle without CSRNET_ALLOW_PICKLE, ...
        error = f"No trained model: counts unavailable. Could not load CSRNet weights {settings.csrnet_weights}: {exc}"
        log.warning(error)
        return Services(settings, by_id, router, store, source, None, error)

    store.model_kind = estimator.info["kind"]  # stored with every reading and alert
    from app.detector import load_detector

    detector, detector_error = load_detector(settings.yolo_weights, settings.device, settings.yolo_imgsz,
                                             settings.yolo_conf)
    if detector_error:
        log.warning(detector_error)
    else:
        log.info("hybrid count: %s below %g people (HYBRID_RULE=%s), CSRNet above", detector.info["model"],
                 settings.hybrid_threshold, settings.hybrid_rule)
    analyzers = {z.id: ZoneAnalyzer(z, estimator, settings, detector=detector) for z in zones}
    dispatcher = AlertDispatcher(build_channels(settings), AlertPolicy(settings.alert_cooldown_s))
    worker = InferenceWorker(zones, analyzers, source, router, dispatcher, store)
    return Services(settings, by_id, router, store, source, worker, gradcam=gradcam, estimator=estimator,
                    detector=detector, detector_error=detector_error)
