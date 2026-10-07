"""Drawing helpers: density heatmaps, ROI outlines and status banners."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from app.risk_engine import RiskLevel

LEVEL_BGR = {
    RiskLevel.GREEN: (80, 175, 76),
    RiskLevel.YELLOW: (40, 215, 250),
    RiskLevel.ORANGE: (0, 140, 255),
    RiskLevel.RED: (50, 50, 230),
}


def density_heatmap(density: np.ndarray, size: tuple[int, int], vmax: float | None = None) -> np.ndarray:
    """Colour-mapped density resized to `size` (w, h). Fix `vmax` to compare frames.

    By default the scale is the map's maximum, but at least 0.05 people per cell, so an
    almost empty scene is not stretched to full red.
    """
    d = np.clip(density, 0, None)
    vmax = vmax or max(float(d.max()), 0.05)
    norm = np.clip(d / vmax * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(cv2.resize(norm, size, interpolation=cv2.INTER_LINEAR), cv2.COLORMAP_JET)


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
