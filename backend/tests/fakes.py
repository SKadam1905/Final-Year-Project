"""Stand-ins for CSRNet and the cameras, so the worker and API run without a model or video."""

from __future__ import annotations

import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from app.alert_dispatcher import AlertDispatcher, AlertPolicy, Delivery
from app.config import ZoneConfig, get_settings
from app.csrnet_model import CSRNet, DensityEstimator, DensityResult
from app.detector import Detections
from app.exit_router import ExitRouter
from app.pipeline import ZoneAnalyzer
from app.store import MemoryStore
from app.worker import InferenceWorker, Services
from ml.gradcam import GradCAM
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


class FakeDetector:
    """`count` person boxes (20 x 40 px) in a row across the frame, or the given `boxes`."""

    info = {"model": "fake-yolo", "imgsz": 640, "conf": 0.25, "device": "cpu", "source": "test"}

    def __init__(self, count: int = 10, boxes=None) -> None:
        self.count = count
        self.boxes = boxes

    def detect(self, frame: np.ndarray) -> Detections:
        if self.boxes is not None:
            boxes = np.asarray(self.boxes, np.float32).reshape(-1, 4)
        else:
            h, w = frame.shape[:2]
            xs = np.linspace(0, w - 20, self.count) if self.count else np.zeros(0)
            boxes = np.array([[x, h / 2 - 40, x + 20, h / 2] for x in xs], np.float32).reshape(-1, 4)
        return Detections(boxes, np.full(len(boxes), 0.9, np.float32), 1.0)


def real_estimator(tmp_path) -> DensityEstimator:
    """A real (randomly initialised) CSRNet on the CPU, for Grad-CAM."""
    torch.manual_seed(0)
    weights = tmp_path / "csrnet.pth"
    torch.save({"model": CSRNet(pretrained_frontend=False).state_dict()}, weights)
    return DensityEstimator(weights, device="cpu", max_width=96)


class FakeSource:
    """A fresh, motionless 5-frame burst for every zone (or none, with frames=None)."""

    def __init__(self, frames: list[np.ndarray] | None = None, empty: bool = False) -> None:
        self.frames = None if empty else frames or [np.dstack([texture(64, 96)] * 3)] * 5
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def next_burst(self, zone_id):
        return (self.frames, time.time()) if self.frames else None

    def peek_burst(self, zone_id):
        return (self.frames, time.time()) if self.frames else None

    def camera_status(self, zone_id):
        return {"status": "online" if self.frames else "connecting"}


class RecordingChannel:
    name = "fake"

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, text):
        self.sent.append(text)
        return [Delivery(self.name, "test", True, 1.0)]


def make_services(count: float = 100.0, store=None, model: bool = True, source=None, estimator=None,
                  detector=None, **overrides) -> Services:
    overrides.setdefault("upload_dir", Path(tempfile.mkdtemp(prefix="kumbh_uploads_")))  # never data/uploads
    overrides.setdefault("demo_mode", True)  # the fake estimator has no provenance; only tests may use it
    settings = replace(get_settings(), **overrides)
    router = ExitRouter()
    store = store or MemoryStore()
    source = source or FakeSource()
    zones = {z.id: z for z in ZONES}
    if not model:
        return Services(settings, zones, router, store, source, None, "CSRNet weights not found")
    estimator = estimator or FakeEstimator(count)
    analyzers = {z.id: ZoneAnalyzer(z, estimator, settings, detector=detector) for z in ZONES}
    channel = RecordingChannel()
    dispatcher = AlertDispatcher([channel], AlertPolicy(settings.alert_cooldown_s))
    worker = InferenceWorker(ZONES, analyzers, source, router, dispatcher, store)
    worker.estimator, worker.channel = estimator, channel  # handles for tests
    gradcam = GradCAM(estimator.model) if isinstance(estimator, DensityEstimator) else None
    return Services(settings, zones, router, store, source, worker, gradcam=gradcam, estimator=estimator,
                    detector=detector, detector_error=None if detector else "detection is off (test)")
