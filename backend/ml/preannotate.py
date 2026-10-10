"""Suggested head points for a crop, so annotators correct points instead of clicking every head (step 4).

Detector and density model each cover what they do well, region by region: every person the
detector finds gets a head point, the density map is cleared inside the detection boxes, and the
density left over (people too small, distant or hidden for the detector) becomes points. Choosing
one method per crop failed both ways on the Kumbh news clips: the detector found 0-12 people in
crops where distant or seated crowds made the density model count 46-56, while on close-ups of
9-12 soldiers the density model counted 22-31.

Density to points: the map (1/8 resolution) is upsampled to the crop, then the highest point is
taken repeatedly and a Gaussian of mass 1, sized to the local head spacing, is subtracted from it,
until round(count) points are placed. The points sit where the model sees people and number what
it counts, but they are suggestions: in dense areas they must be moved onto the actual heads, and
the public weights undercount dense, distant crowds, so expect to add heads there.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_POINTS = 3000  # a 512x512 crop holding more heads than this is beyond the suggestions' usefulness


@dataclass(frozen=True)
class Suggestion:
    points: np.ndarray  # (N, 2) x, y in crop pixels
    method: str  # "detector" or "density"
    density_count: float
    detector_count: int | None  # None when no detector is loaded


def heads_from_boxes(boxes: np.ndarray) -> np.ndarray:
    """Head centre of each person box: horizontally centred, about a fifth of the box width below its top.

    A head is roughly a third of the shoulder width, so its centre sits ~0.2 x width under the
    box top; capped at mid-height for boxes cut off by the crop edge or seated people.
    """
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 1] + np.minimum(0.2 * w, 0.5 * h)])


def points_from_density(density: np.ndarray, size: tuple[int, int], count: float | None = None) -> np.ndarray:
    """round(count) points (default: the map's sum) placed at the density's peaks, in pixels of `size` (w, h)."""
    w, h = size
    total = float(density.sum()) if count is None else float(count)
    n = min(MAX_POINTS, int(round(total)))
    if n <= 0:
        return np.zeros((0, 2))
    residual = cv2.resize(density.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    residual = np.maximum(residual, 0)
    residual *= total / max(float(residual.sum()), 1e-9)  # heads per pixel, summing to the count
    points = []
    for _ in range(n):
        y, x = divmod(int(np.argmax(residual)), w)
        peak = float(residual[y, x])
        if peak <= 0:
            break
        sigma = float(np.clip(1.0 / np.sqrt(peak) / 2.5, 1.5, 12.0))  # head spacing 1/sqrt(density)
        r = int(np.ceil(3 * sigma))
        y0, y1, x0, x1 = max(0, y - r), min(h, y + r + 1), max(0, x - r), min(w, x + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        kernel = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        residual[y0:y1, x0:x1] = np.maximum(residual[y0:y1, x0:x1] - kernel, 0)
        points.append((x + 0.5, y + 0.5))
    return np.asarray(points, np.float64).reshape(-1, 2)


def suggest_points(crop_bgr: np.ndarray, estimator, detector=None) -> Suggestion:
    h, w = crop_bgr.shape[:2]
    result = estimator.predict(crop_bgr)
    if detector is None:
        return Suggestion(points_from_density(result.density, (w, h)), "density", result.count, None)
    detections = detector.detect(crop_bgr)
    density = result.density.copy()
    sx, sy = density.shape[1] / w, density.shape[0] / h
    for x1, y1, x2, y2 in detections.boxes:  # cells whose centre lies in the box
        density[round(y1 * sy):round(y2 * sy), round(x1 * sx):round(x2 * sx)] = 0
    heads = heads_from_boxes(detections.boxes)
    rest = points_from_density(density, (w, h))
    method = "detector+density" if len(heads) and len(rest) else "detector" if len(heads) else "density"
    return Suggestion(np.vstack([heads, rest]), method, result.count, detections.count)
