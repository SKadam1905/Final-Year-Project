"""Optional tiled / upscaled inference for small, distant heads (counting-accuracy experiment).

Not used by the live pipeline: scripts/kumbh_eval_tiling.py measures it on the Kumbh evaluation
frames, and nothing changes in how the system counts unless it is wired in later.

* YOLO, SAHI-style slicing (Akyon et al. 2022): the frame is cut into square tiles (default 320 px,
  20% overlap), each tile is upscaled to the detector's input size (640 px, i.e. 2x) and detected
  on, the whole frame is detected on as well (for people larger than a tile), and the boxes are
  mapped back to frame pixels and merged. Inside one tile YOLO has already removed its own
  duplicates, so merging only compares boxes from different tiles (or tile vs whole frame): the
  lower-scoring box is dropped when its intersection covers at least half of the smaller box
  (intersection over smaller, SAHI's metric for sliced predictions), which also removes the
  partial box of a person cut by a tile edge. People overlapping inside one tile are kept.
* CSRNet on the frame upscaled 2x or 3x (bicubic): heads become larger than the few pixels CSRNet
  can resolve. The density map then covers the upscaled frame; it still sums to the number of
  people, and counting inside a region maps its cells back to original pixels
  (ml.kumbh_eval.density_in_region takes the original frame size).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np

from app.detector import PERSON, Detections

BoxesScores = tuple[np.ndarray, np.ndarray]  # (N, 4) x1, y1, x2, y2 and (N,) scores


def tile_starts(length: int, tile: int, overlap: float) -> list[int]:
    """Start offsets of tiles covering [0, length): step tile * (1 - overlap), the last one flush with the end."""
    if length <= tile:
        return [0]
    step = max(1, round(tile * (1 - overlap)))
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tiles(width: int, height: int, tile: int = 320, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) of every tile, row by row; tiles are clipped to frames smaller than `tile`."""
    tw, th = min(tile, width), min(tile, height)
    return [(x, y, tw, th) for y in tile_starts(height, tile, overlap) for x in tile_starts(width, tile, overlap)]


def merge_across_sources(boxes: np.ndarray, scores: np.ndarray, sources: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Indices kept by greedy, score-ordered suppression by intersection over the smaller box, only
    between boxes from different sources (tiles / whole frame)."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores), kind="stable")
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    keep: list[int] = []
    for i in order:
        if keep:
            k = np.asarray(keep)
            iw = np.minimum(boxes[k, 2], boxes[i, 2]) - np.maximum(boxes[k, 0], boxes[i, 0])
            ih = np.minimum(boxes[k, 3], boxes[i, 3]) - np.maximum(boxes[k, 1], boxes[i, 1])
            inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
            ios = inter / np.maximum(np.minimum(areas[k], areas[i]), 1e-9)
            if np.any((ios >= threshold) & (sources[k] != sources[i])):
                continue
        keep.append(int(i))
    return np.asarray(sorted(keep), int)


def sliced_detect(detect_batch: Callable[[list[np.ndarray]], list[BoxesScores]], frame: np.ndarray,
                  whole: BoxesScores | None = None, tile: int = 320, overlap: float = 0.2, upscale_to: int = 640,
                  threshold: float = 0.5) -> Detections:
    """SAHI-style detection: tiles upscaled to `upscale_to`, plus the whole-frame boxes `whole`, merged."""
    started = time.perf_counter()
    h, w = frame.shape[:2]
    rects = tiles(w, h, tile, overlap)
    crops = [cv2.resize(frame[y:y + th, x:x + tw], (round(tw * upscale_to / max(tw, th)), round(th * upscale_to / max(tw, th))),
                        interpolation=cv2.INTER_CUBIC) for x, y, tw, th in rects]
    all_boxes, all_scores, all_sources = [], [], []
    for k, ((x, y, tw, th), (boxes, scores)) in enumerate(zip(rects, detect_batch(crops))):
        scale = max(tw, th) / upscale_to
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4) * scale + [x, y, x, y]
        all_boxes.append(boxes)
        all_scores.append(np.asarray(scores, np.float64).reshape(-1))
        all_sources.append(np.full(len(boxes), k))
    if whole is not None:
        all_boxes.append(np.asarray(whole[0], np.float64).reshape(-1, 4))
        all_scores.append(np.asarray(whole[1], np.float64).reshape(-1))
        all_sources.append(np.full(len(all_boxes[-1]), -1))
    boxes, scores, sources = np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_sources)
    keep = merge_across_sources(boxes, scores, sources, threshold)
    return Detections(boxes[keep].astype(np.float32).reshape(-1, 4), scores[keep].astype(np.float32),
                      (time.perf_counter() - started) * 1e3)


def yolo_batch(detector) -> Callable[[list[np.ndarray]], list[BoxesScores]]:
    """Batched person detection with an app.detector.PersonDetector's model and settings."""
    def run(images: Sequence[np.ndarray]) -> list[BoxesScores]:
        with detector._lock:
            results = detector.model.predict(list(images), imgsz=detector.imgsz, conf=detector.conf, classes=[PERSON],
                                             device=detector.device, quantize=detector.quantize, verbose=False)
        return [(r.boxes.xyxy.cpu().numpy().reshape(-1, 4), r.boxes.conf.cpu().numpy().reshape(-1)) for r in results]
    return run


@dataclass(frozen=True)
class UpscaledDensity:
    density: np.ndarray  # map over the upscaled frame (1/8 of its size); sums to `count`
    count: float
    ms: float


def upscaled_density(estimator, frame: np.ndarray, factor: float) -> UpscaledDensity:
    """CSRNet on the frame upscaled by `factor` (bicubic). The estimator must not downscale it back:
    give it max_width=None (or at least the upscaled width)."""
    started = time.perf_counter()
    h, w = frame.shape[:2]
    big = frame if factor == 1 else cv2.resize(frame, (round(w * factor), round(h * factor)),
                                                interpolation=cv2.INTER_CUBIC)
    result = estimator.predict(big)
    return UpscaledDensity(result.density, result.count, (time.perf_counter() - started) * 1e3)
