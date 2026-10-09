"""Drawing helpers: density heatmaps, ROI outlines and status banners.

Density heatmaps shown to people (live overlay, video report, annotated MP4) use one fixed
colour scale in persons/m² (DensityScale), the same for every frame and zone and aligned with
the risk thresholds, so a colour always means the same crowding. normalise_density /
density_heatmap / overlay_density scale each map to its own maximum and are only for
model-evaluation figures, where no ground area is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from app.risk_engine import RiskLevel

DENSITY_UNIT = "persons/m²"


@dataclass(frozen=True)
class DensityScale:
    """Fixed persons/m² -> colour scale. Colours are interpolated between the stops; values
    above the last stop keep its colour."""

    stops: tuple[tuple[float, str], ...]  # (persons/m², "#rrggbb"), increasing

    @property
    def max(self) -> float:
        return self.stops[-1][0]

    def lut(self) -> np.ndarray:
        """(256, 3) uint8 BGR colours for values 0..max."""
        values = np.linspace(0.0, self.max, 256)
        xs = [v for v, _ in self.stops]
        rgb = np.array([[int(c[i : i + 2], 16) for i in (1, 3, 5)] for _, c in self.stops], np.float64)
        channels = [np.interp(values, xs, rgb[:, k]) for k in (2, 1, 0)]
        return np.clip(np.round(np.stack(channels, axis=1)), 0, 255).astype(np.uint8)

    def to_dict(self) -> dict:
        return {"unit": DENSITY_UNIT, "max": self.max, "stops": [{"value": v, "colour": c} for v, c in self.stops]}


def density_scale(yellow: float = 2.0, orange: float = 4.0, red: float = 6.0) -> DensityScale:
    """Green at 0, yellow at the YELLOW threshold, orange at ORANGE, red at RED, dark red one above."""
    return DensityScale(((0.0, "#15803d"), (yellow, "#facc15"), (orange, "#f97316"), (red, "#b91c1c"),
                         (red + 1.0, "#7f1d1d")))


def density_rgba(per_m2: np.ndarray, size: tuple[int, int], scale: DensityScale, max_alpha: float = 0.75,
                 smooth_cells: float = 1.5) -> np.ndarray:
    """Transparent BGRA overlay of `size` (w, h) for a map of persons/m² per density cell.

    Fixed scale: colour and opacity depend only on persons/m², never on the frame's own maximum,
    so a sparse scene stays faint green. The map is lightly smoothed (Gaussian, `smooth_cells`)
    so the colour shows local density rather than single-cell noise.
    """
    v = np.nan_to_num(np.asarray(per_m2, np.float32), nan=0.0, posinf=0.0)
    v = np.clip(v, 0.0, None)
    if smooth_cells:
        v = cv2.GaussianBlur(v, (0, 0), smooth_cells)
    v = cv2.resize(v, size, interpolation=cv2.INTER_LINEAR)
    fraction = np.clip(v / scale.max, 0.0, 1.0)
    bgr = scale.lut()[np.round(fraction * 255).astype(np.uint8)]
    alpha = (np.sqrt(fraction) * max_alpha * 255).astype(np.uint8)
    return np.dstack([bgr, alpha])


def blend_rgba(frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Composite a BGRA overlay of the frame's size onto a BGR frame."""
    alpha = overlay[..., 3:4].astype(np.float32) / 255.0
    return (frame.astype(np.float32) * (1 - alpha) + overlay[..., :3].astype(np.float32) * alpha).astype(np.uint8)


def overlay_density_m2(frame: np.ndarray, per_m2: np.ndarray, scale: DensityScale) -> np.ndarray:
    """The frame with its persons/m² heatmap on the fixed scale (for annotated videos)."""
    h, w = frame.shape[:2]
    return blend_rgba(frame, density_rgba(per_m2, (w, h), scale))


def draw_density_legend(frame: np.ndarray, scale: DensityScale) -> np.ndarray:
    """A small colour bar with its persons/m² ticks in the bottom-right corner (in place)."""
    h, w = frame.shape[:2]
    s = max(0.6, w / 1280)
    bar_w, bar_h, pad = int(220 * s), int(12 * s), int(10 * s)
    if w < bar_w + 3 * pad or h < 8 * bar_h:
        return frame  # too small to carry a legend
    x0, y0 = w - bar_w - pad, h - bar_h - int(34 * s)
    box = frame[y0 - int(22 * s) : h - int(4 * s), x0 - pad : w - int(4 * s)]
    box[:] = (box * 0.35).astype(np.uint8)  # dark backing so the bar reads on any scene
    frame[y0 : y0 + bar_h, x0 : x0 + bar_w] = cv2.resize(scale.lut()[None], (bar_w, bar_h), interpolation=cv2.INTER_LINEAR)
    font, size, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.4 * s, max(1, int(s))
    cv2.putText(frame, DENSITY_UNIT.replace("²", "2"), (x0, y0 - int(6 * s)), font, size, (255, 255, 255), thick, cv2.LINE_AA)
    for value, _ in scale.stops:
        x = x0 + int(bar_w * value / scale.max)
        label = f"{value:g}+" if value == scale.max else f"{value:g}"
        cv2.putText(frame, label, (min(x, x0 + bar_w - int(14 * s)), y0 + bar_h + int(14 * s)), font, size,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return frame

LEVEL_BGR = {
    RiskLevel.GREEN: (80, 175, 76),
    RiskLevel.YELLOW: (40, 215, 250),
    RiskLevel.ORANGE: (0, 140, 255),
    RiskLevel.RED: (50, 50, 230),
}


def normalise_density(density: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Density scaled to 0..1. Fix `vmax` to compare frames.

    By default the scale is the map's maximum, but at least 0.05 people per cell, so an
    almost empty scene is not stretched to full red.
    """
    d = np.clip(density, 0, None)
    return np.clip(d / (vmax or max(float(d.max()), 0.05)), 0.0, 1.0)


def density_heatmap(density: np.ndarray, size: tuple[int, int], vmax: float | None = None) -> np.ndarray:
    """Colour-mapped density resized to `size` (w, h)."""
    norm = (normalise_density(density, vmax) * 255).astype(np.uint8)
    return cv2.applyColorMap(cv2.resize(norm, size, interpolation=cv2.INTER_LINEAR), cv2.COLORMAP_JET)


def colour_overlay(values: np.ndarray, size: tuple[int, int], max_alpha: float = 0.75) -> np.ndarray:
    """Transparent BGRA overlay of `size` (w, h) for values in 0..1 (a density or Grad-CAM map).

    TURBO colours; opacity rises with the value, so low areas leave the camera image visible.
    """
    v = cv2.resize(np.clip(values, 0.0, 1.0).astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    bgr = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = (np.sqrt(v) * max_alpha * 255).astype(np.uint8)
    return np.dstack([bgr, alpha])


def overlay_density(frame: np.ndarray, density: np.ndarray, alpha: float = 0.45, vmax: float | None = None) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.addWeighted(frame, 1 - alpha, density_heatmap(density, (w, h), vmax), alpha, 0)


def draw_polygon(frame: np.ndarray, polygon: Sequence[tuple[float, float]], color=(255, 255, 255)) -> np.ndarray:
    pts = np.round(np.asarray(polygon)).astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    return frame


def draw_banner(frame: np.ndarray, lines: Sequence[str], level: RiskLevel | None = None) -> np.ndarray:
    """Status text on a band across the top, coloured by risk level."""
    scale = max(0.5, frame.shape[1] / 1600)
    line_h = int(32 * scale)
    band_h = line_h * len(lines) + int(12 * scale)
    color = LEVEL_BGR.get(level, (60, 60, 60))
    band = frame[:band_h].copy()
    cv2.rectangle(band, (0, 0), (frame.shape[1], band_h), color, -1)
    frame[:band_h] = cv2.addWeighted(band, 0.75, frame[:band_h], 0.25, 0)
    text_color = (0, 0, 0) if level in (RiskLevel.YELLOW, RiskLevel.ORANGE) else (255, 255, 255)
    for i, line in enumerate(lines):
        y = int(8 * scale) + line_h * (i + 1) - int(8 * scale)
        cv2.putText(frame, line, (int(12 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, text_color,
                    max(1, int(2 * scale)), cv2.LINE_AA)
    return frame


DETECTION_BGR = (238, 211, 34)  # cyan (#22d3ee): stands out on crowds and on the heatmap colours


def detections_rgba(boxes: np.ndarray | None, size: tuple[int, int], frame_size: tuple[int, int]) -> np.ndarray:
    """Transparent BGRA overlay of `size` (w, h) with a box around each detected person.

    `boxes` (N, 4) are x1, y1, x2, y2 in the pixels of a (w, h) `frame_size` frame; they are scaled
    to `size`, so the overlay lines up with a resized snapshot.
    """
    w, h = size
    out = np.zeros((h, w, 4), np.uint8)
    if boxes is None or len(boxes) == 0:
        return out
    sx, sy = w / frame_size[0], h / frame_size[1]
    thickness = max(1, round(w / 640))
    for x1, y1, x2, y2 in np.asarray(boxes, np.float64).reshape(-1, 4):
        cv2.rectangle(out, (round(x1 * sx), round(y1 * sy)), (round(x2 * sx), round(y2 * sy)),
                      (*DETECTION_BGR, 255), thickness)
    return out
