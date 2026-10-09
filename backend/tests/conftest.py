import os
from pathlib import Path

import cv2
import numpy as np
import pytest

# Tests never load YOLO from .env's weights (slow, needs the GPU); detector tests use fakes.
# Set before app.config reads the environment (load_dotenv does not override it).
os.environ["YOLO_WEIGHTS"] = "off"
os.environ["SECRET_KEY"] = "pytest-secret-key"  # session tokens (app.auth); never the one in .env


def write_video(path: Path, frames: list[np.ndarray], fps: float = 25.0) -> Path:
    """Write frames to an MJPG .avi (a codec every OpenCV build can read and write)."""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert writer.isOpened(), "OpenCV cannot write MJPG video"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def texture(h: int = 240, w: int = 320, seed: int = 0, blur: float = 2.0) -> np.ndarray:
    """Smooth random grey texture that optical flow can track."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, (h, w)).astype(np.float32)
    return cv2.normalize(cv2.GaussianBlur(noise, (0, 0), blur), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
