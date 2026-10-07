"""Image read/write that also works with non-ASCII Windows paths (cv2.imread does not)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Read a colour image as BGR uint8."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray, quality: int = 95) -> None:
    path = Path(path)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"cannot encode image as {ext}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(path))
