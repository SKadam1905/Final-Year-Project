"""Stand-ins for CSRNet and the cameras, so the worker and API run without a model or video."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from app.alert_dispatcher import AlertDispatcher, AlertPolicy, Delivery
from app.config import ZoneConfig, get_settings
from app.csrnet_model import DensityResult
from app.exit_router import ExitRouter
from app.pipeline import ZoneAnalyzer
from app.store import MemoryStore
from app.worker import InferenceWorker, Services
from tests.conftest import texture

ZONES = [
    ZoneConfig("RAMKUND_NORTH", "Ramkund north steps", "rtsp://admin:secret@10.0.0.5:554/stream1", area_m2=100.0),
    ZoneConfig("CAMERA_X", "Off-graph camera", "rtsp://10.0.0.6/stream", area_m2=100.0),
]


class FakeEstimator:
    """A uniform density map whose total is `count` people (set per test)."""

    def __init__(self, count: float = 100.0) -> None:
        self.count = count

    def predict(self, frame: np.ndarray) -> DensityResult:
        h, w = frame.shape[:2]
        density = np.full((h // 8, w // 8), self.count / ((h // 8) * (w // 8)), np.float32)
        return DensityResult(density, float(density.sum()), (w, h), (w, h))


class FakeSource:
    """Always has a fresh, motionless 5-frame burst for every zone."""

    def __init__(self) -> None:
        self.frames = [np.dstack([texture(64, 96)] * 3)] * 5
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def next_burst(self, zone_id):
        return self.frames, time.time()

    def peek_burst(self, zone_id):
        return self.frames, time.time()

    def camera_status(self, zone_id):
        return {"status": "online"}


class RecordingChannel:
    name = "fake"

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, text):
        self.sent.append(text)
        return [Delivery(self.name, "test", True, 1.0)]


def make_services(count: float = 100.0, store=None, model: bool = True, **overrides) -> Services:
    settings = replace(get_settings(), **overrides)
    router = ExitRouter()
    store = store or MemoryStore()
    source = FakeSource()
    zones = {z.id: z for z in ZONES}
    if not model:
        return Services(settings, zones, router, store, source, None, "CSRNet weights not found")
    estimator = FakeEstimator(count)
    analyzers = {z.id: ZoneAnalyzer(z, estimator, settings) for z in ZONES}
    channel = RecordingChannel()
    dispatcher = AlertDispatcher([channel], AlertPolicy(settings.alert_cooldown_s))
    worker = InferenceWorker(ZONES, analyzers, source, router, dispatcher, store)
    worker.estimator, worker.channel = estimator, channel  # handles for tests
    return Services(settings, zones, router, store, source, worker)
