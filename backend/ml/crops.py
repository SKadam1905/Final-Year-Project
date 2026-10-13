"""Crops centred on the densest crowd areas, and contact sheets for reviewing them (step 3).

A frame is first brought to the working resolution the live system uses (frames wider than
INFER_MAX_WIDTH are downscaled to it), so heads in the crops have the size the model sees at
deployment. The density model's map (1/8 resolution) gives, through an integral image, the
predicted count inside every crop-sized window; the best window is taken, windows overlapping
it are ruled out, and a second crop is kept only if it is reasonably crowded too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.csrnet_model import OUTPUT_STRIDE


@dataclass(frozen=True)
class CropBox:
    x: int  # top-left, in working-resolution pixels
    y: int
    size: int
    predicted: float  # density-model count inside the crop


def crop_size(width: int, height: int, wanted: int = 512) -> int:
    """`wanted`, or the frame's short side if smaller; a multiple of the model's stride."""
    return min(wanted, width, height) // OUTPUT_STRIDE * OUTPUT_STRIDE


def densest_crops(density: np.ndarray, size: int, max_crops: int = 2, exclude: list[tuple[int, int, int, int]] = (),
                  min_ratio: float = 0.3, min_count: float = 2.0) -> list[CropBox]:
    """Up to `max_crops` non-overlapping size x size windows with the highest predicted count.

    `density` is the model output at 1/8 of the working resolution; `exclude` holds boxes
    (x1, y1, x2, y2, working pixels) a crop must not touch, e.g. channel logos. A crop after
    the first is kept only if its count is at least `min_ratio` of the first's and `min_count`.
    """
    cells = size // OUTPUT_STRIDE
    gh, gw = density.shape
    if cells < 1 or cells > gh or cells > gw:
        return []
    integral = cv2.integral(density.astype(np.float64))
    sums = integral[cells:, cells:] - integral[:-cells, cells:] - integral[cells:, :-cells] + integral[:-cells, :-cells]
    valid = np.ones_like(sums, bool)  # sums[j, i] = window with its top-left cell at (i, j)
    for x1, y1, x2, y2 in exclude:  # windows touching the box: cells i with i*8 < x2 and i*8 + size > x1
        i0, i1 = max(0, math.floor((x1 - size) / OUTPUT_STRIDE) + 1), min(sums.shape[1], math.ceil(x2 / OUTPUT_STRIDE))
        j0, j1 = max(0, math.floor((y1 - size) / OUTPUT_STRIDE) + 1), min(sums.shape[0], math.ceil(y2 / OUTPUT_STRIDE))
        valid[j0:j1, i0:i1] = False
    crops: list[CropBox] = []
    while len(crops) < max_crops and valid.any():
        scores = np.where(valid, sums, -np.inf)
        j, i = np.unravel_index(int(np.argmax(scores)), scores.shape)
        count = float(sums[j, i])
        if crops and count < max(min_count, min_ratio * crops[0].predicted):
            break
        crops.append(CropBox(int(i) * OUTPUT_STRIDE, int(j) * OUTPUT_STRIDE, size, count))
        valid[max(0, j - cells + 1):j + cells, max(0, i - cells + 1):i + cells] = False  # no overlap
    return crops


def contact_sheet(tiles: list[tuple[np.ndarray, str]], columns: int = 5, tile: int = 256) -> np.ndarray:
    """Labelled thumbnails in a grid (BGR); the label is printed on a dark bar at the top."""
    cells = []
    for image, label in tiles:
        thumb = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (tile, 20), (0, 0, 0), -1)
        cv2.putText(thumb, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cells.append(cv2.copyMakeBorder(thumb, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
    if not cells:
        return np.zeros((tile, tile, 3), np.uint8)
    while len(cells) % columns:
        cells.append(np.zeros_like(cells[0]))
    return np.vstack([np.hstack(cells[r:r + columns]) for r in range(0, len(cells), columns)])
