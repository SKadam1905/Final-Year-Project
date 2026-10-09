"""YOLOv8 person detection for the hybrid count (COCO-pretrained weights, class 0 "person" only).

Density maps (CSRNet) are made for dense crowds, where people hide each other; a detector is more
accurate when a scene is sparse and people are fully visible. The hybrid rule (app.pipeline):
when YOLO finds fewer than HYBRID_THRESHOLD people (default 60) its count is used ("detection"),
otherwise CSRNet's ("density").

    YOLO_WEIGHTS=backend/ml/weights/yolo/yolov8s.pt    # downloaded by tools.get_yolo_weights
    YOLO_WEIGHTS=off                                   # density only (no detection, no hybrid)

The ultralytics package and its weights are optional: without them the system counts with CSRNet
only and says so in /health.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.csrnet_model import resolve_device

log = logging.getLogger(__name__)

PERSON = 0  # COCO class id


@dataclass(frozen=True)
class Detections:
    boxes: np.ndarray  # (N, 4) float32 x1, y1, x2, y2 in frame pixels
    scores: np.ndarray  # (N,) float32 confidences
    ms: float  # detection time

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def feet(self) -> np.ndarray:
        """(N, 2) bottom-centre of each box: where the person stands, for ROI tests and grid cells."""
        return np.column_stack([(self.boxes[:, 0] + self.boxes[:, 2]) / 2, self.boxes[:, 3]]).astype(np.float32)

    def to_list(self) -> list[list[float]]:
        """Boxes with scores, rounded, for JSON: [[x1, y1, x2, y2, score], ...]."""
        return [[round(float(v), 1) for v in box] + [round(float(s), 3)] for box, s in zip(self.boxes, self.scores)]


def empty_detections() -> Detections:
    return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.0)


class PersonDetector:
    """Runs a YOLOv8 model on single BGR frames and keeps the person boxes."""

    def __init__(self, weights: str | Path, device: str = "auto", imgsz: int = 640, conf: float = 0.25) -> None:
        from ultralytics import YOLO

        self.device = resolve_device(device)
        self.model = YOLO(str(weights), task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.quantize = 16 if self.device.type == "cuda" else None  # FP16 on the GPU
        self.info = {"model": Path(weights).stem, "imgsz": imgsz, "conf": conf, "device": str(self.device),
                     "source": "ultralytics COCO-pretrained (person class only)"}
        self._lock = threading.Lock()  # the ultralytics predictor is not thread-safe

    def detect(self, frame_bgr: np.ndarray) -> Detections:
        started = time.perf_counter()
        with self._lock:
            result = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, classes=[PERSON],
                                        device=self.device, quantize=self.quantize, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32).reshape(-1, 4)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1)
        return Detections(boxes, scores, (time.perf_counter() - started) * 1e3)


def load_detector(weights: Path | None, device: str = "auto", imgsz: int = 640,
                  conf: float = 0.25) -> tuple[PersonDetector | None, str | None]:
    """(detector, None), or (None, why not) when detection is off or unavailable."""
    if weights is None:
        return None, "detection is off (YOLO_WEIGHTS=off): counts come from CSRNet only"
    if not weights.exists():
        return None, (f"YOLO weights not found at {weights}: counts come from CSRNet only. Download them with: "
                      f"cd backend; ..\\venv\\Scripts\\python.exe -m tools.get_yolo_weights {weights.stem}")
    try:
        detector = PersonDetector(weights, device, imgsz, conf)
        detector.detect(np.zeros((imgsz // 2, imgsz, 3), np.uint8))  # warm-up: first call builds the predictor
    except Exception as exc:  # ultralytics missing or a broken file
        return None, f"YOLO person detection unavailable ({exc}): counts come from CSRNet only"
    return detector, None
