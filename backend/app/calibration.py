"""Camera calibration: convert a density map into persons per square metre (plan §4.5).

A homography H maps image pixels on the ground plane to ground coordinates in metres. It
is estimated from >= 4 floor markers measured on site. Each density-map cell covers a
different ground area (far cells cover more), so the zone density is

    (heads in the zone's cells) / (ground area of the same cells)

A homography only describes the ground plane: people on steps or risers add error, which
belongs in the thesis limitations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.config import ZoneConfig

Point = tuple[float, float]


def compute_homography(image_points: Sequence[Point], ground_points: Sequence[Point]) -> np.ndarray:
    """Image-pixel -> ground-metre homography from >= 4 floor markers (least squares)."""
    src = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(ground_points, dtype=np.float64).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("need the same number (>= 4) of image and ground points")
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None or not np.isfinite(H).all():
        raise ValueError("calibration points are degenerate (are three of them collinear?)")
    # Scale H so the projective denominator is +1 at a marker, i.e. positive on the ground
    # in front of the camera and negative beyond the horizon.
    return H / (H[2] @ np.array([src[0, 0], src[0, 1], 1.0]))


def project(H: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map image points (N, 2) to ground points (N, 2); also return the denominators."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    homog = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    w = homog[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return homog[:, :2] / w[:, None], w


def cell_ground_areas(H: np.ndarray, frame_size: tuple[int, int], grid_shape: tuple[int, int]) -> np.ndarray:
    """Ground area in m² of each cell of a (rows, cols) grid laid over a (w, h) frame.

    Cells touching or above the horizon get NaN.
    """
    frame_w, frame_h = frame_size
    rows, cols = grid_shape
    gx, gy = np.meshgrid(np.linspace(0.0, frame_w, cols + 1), np.linspace(0.0, frame_h, rows + 1))
    ground, w = project(H, np.column_stack([gx.ravel(), gy.ravel()]))
    x = ground[:, 0].reshape(rows + 1, cols + 1)
    y = ground[:, 1].reshape(rows + 1, cols + 1)
    ok = (w > 1e-12).reshape(rows + 1, cols + 1)

    # Cell corners in order: top-left, top-right, bottom-right, bottom-left.
    corners = [(slice(None, -1), slice(None, -1)), (slice(None, -1), slice(1, None)),
               (slice(1, None), slice(1, None)), (slice(1, None), slice(None, -1))]
    cx = [x[c] for c in corners]
    cy = [y[c] for c in corners]
    with np.errstate(invalid="ignore"):
        twice_area = sum(cx[i] * cy[(i + 1) % 4] - cx[(i + 1) % 4] * cy[i] for i in range(4))
    areas = 0.5 * np.abs(twice_area)
    valid = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, 1:] & ok[1:, :-1]
    areas[~valid] = np.nan
    return areas


def points_in_polygon(points: np.ndarray, polygon: Sequence[Point]) -> np.ndarray:
    """Even-odd rule point-in-polygon test for an (N, 2) array of points."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    poly = np.asarray(polygon, np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    j = len(poly) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(poly)):
            (xi, yi), (xj, yj) = poly[i], poly[j]
            crosses = (yi > y) != (yj > y)
            inside ^= crosses & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            j = i
    return inside


def roi_mask(polygon: Sequence[Point] | None, frame_size: tuple[int, int], grid_shape: tuple[int, int]) -> np.ndarray:
    """Grid cells whose centres lie inside the ROI polygon (all cells if no polygon)."""
    rows, cols = grid_shape
    if polygon is None:
        return np.ones(grid_shape, bool)
    frame_w, frame_h = frame_size
    gx, gy = np.meshgrid((np.arange(cols) + 0.5) * frame_w / cols, (np.arange(rows) + 0.5) * frame_h / rows)
    return points_in_polygon(np.column_stack([gx.ravel(), gy.ravel()]), polygon).reshape(rows, cols)


class ZoneGeometry:
    """Counts heads inside a zone's ROI and divides by its ground area.

    With a homography the area comes from calibration; otherwise from the configured
    `area_m2` of the ROI (a rough on-site estimate).
    """

    def __init__(
        self,
        roi: Sequence[Point] | None = None,
        homography: np.ndarray | None = None,
        area_m2: float | None = None,
    ) -> None:
        if homography is None and not area_m2:
            raise ValueError("need a homography or a positive area_m2")
        self.roi = roi
        self.homography = homography
        self.area_m2 = area_m2
        self._key: tuple | None = None
        self._mask: np.ndarray | None = None
        self._areas: np.ndarray | None = None

    @classmethod
    def from_zone(cls, zone: ZoneConfig) -> ZoneGeometry:
        cal = zone.calibration
        H = compute_homography(cal.image_points, cal.ground_points) if cal else None
        return cls(zone.roi, H, zone.area_m2)

    def _prepare(self, frame_size: tuple[int, int], grid_shape: tuple[int, int]) -> None:
        key = (tuple(frame_size), tuple(grid_shape))
        if key == self._key:
            return
        self._mask = roi_mask(self.roi, frame_size, grid_shape)
        if self.homography is not None:
            self._areas = cell_ground_areas(self.homography, frame_size, grid_shape)
            self._mask &= np.isfinite(self._areas)
        self._key = key

    def cell_density(self, density: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
        """Persons per m² in each density-map cell (0 outside the zone), for heatmaps.

        With calibration each cell is divided by its own ground area. With only `area_m2`, the
        zone's area is shared equally by its cells, so a uniform map shows the zone's density
        everywhere.
        """
        self._prepare(frame_size, density.shape)
        out = np.zeros(density.shape, np.float32)
        if self._areas is not None:
            out[self._mask] = density[self._mask] / self._areas[self._mask]
        else:
            out[self._mask] = density[self._mask] * float(self._mask.sum()) / float(self.area_m2)
        return np.clip(out, 0.0, None)

    def ground_area(self, frame_size: tuple[int, int], grid_shape: tuple[int, int]) -> float:
        """Ground area of the zone in m² (from calibration over a density map's grid, else area_m2)."""
        self._prepare(frame_size, grid_shape)
        area = float(self._areas[self._mask].sum()) if self._areas is not None else float(self.area_m2)
        if area <= 0:
            raise ValueError("zone has no valid ground area; check the ROI and calibration")
        return area

    def measure(self, density: np.ndarray, frame_size: tuple[int, int]) -> tuple[float, float]:
        """(heads in zone, persons per m²) for a density map covering a (w, h) frame."""
        area = self.ground_area(frame_size, density.shape)
        count = float(density[self._mask].sum())
        return count, max(count, 0.0) / area

    def in_zone(self, points: np.ndarray) -> np.ndarray:
        """Which image points (N, 2), e.g. detected people's feet, lie inside the ROI (all without one)."""
        points = np.asarray(points, np.float64).reshape(-1, 2)
        return np.ones(len(points), bool) if self.roi is None else points_in_polygon(points, self.roi)

    def grid(self, density: np.ndarray, frame_size: tuple[int, int], rows: int = 3, cols: int = 3,
             points: np.ndarray | None = None) -> ZoneGrid:
        """Zone-wise segmentation: people and persons/m² in each cell of a rows x cols grid over the
        frame. People come from `points` (detected feet) when given, else from the density map.
        A cell's ground area is the calibrated area of its part of the zone, else its share of area_m2."""
        self._prepare(frame_size, density.shape)
        dh, dw = density.shape
        r = np.minimum(((np.arange(dh) + 0.5) * rows / dh).astype(int), rows - 1)
        c = np.minimum(((np.arange(dw) + 0.5) * cols / dw).astype(int), cols - 1)
        cell = (r[:, None] * cols + c[None, :])[self._mask]  # grid cell of each density-map cell in the zone
        n = rows * cols
        if self._areas is not None:
            areas = np.bincount(cell, weights=self._areas[self._mask], minlength=n)
        else:
            areas = np.bincount(cell, minlength=n) * float(self.area_m2) / max(len(cell), 1)
        if points is None:
            counts = np.bincount(cell, weights=density[self._mask], minlength=n)
        else:
            pts = np.asarray(points, np.float64).reshape(-1, 2)
            pts = pts[self.in_zone(pts)]
            w, h = frame_size
            pr = np.clip((pts[:, 1] * rows / h).astype(int), 0, rows - 1)
            pc = np.clip((pts[:, 0] * cols / w).astype(int), 0, cols - 1)
            counts = np.bincount(pr * cols + pc, minlength=n).astype(np.float64)
        counts = np.clip(counts, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_m2 = np.where(areas > 0, counts / np.where(areas > 0, areas, 1.0), 0.0)
        return ZoneGrid(counts.reshape(rows, cols), per_m2.reshape(rows, cols))


ROW_NAMES = ("top", "middle", "bottom")
COL_NAMES = ("left", "centre", "right")


def cell_label(row: int, col: int, rows: int, cols: int) -> str:
    """'top-left' ... 'centre' ... 'bottom-right' on a 3x3 grid; 'row r, column c' otherwise."""
    if (rows, cols) == (3, 3):
        return "centre" if (row, col) == (1, 1) else f"{ROW_NAMES[row]}-{COL_NAMES[col]}"
    return f"row {row + 1}, column {col + 1}"


class ZoneGrid:
    """People and persons/m² per cell of a camera view's grid, and its most crowded cell."""

    def __init__(self, counts: np.ndarray, density: np.ndarray) -> None:
        self.counts = counts
        self.density = density

    def hotspot(self) -> dict | None:
        """The cell with the highest persons/m² (None if the view is empty)."""
        if self.density.size == 0 or float(self.density.max()) <= 0:
            return None
        row, col = (int(i) for i in np.unravel_index(int(np.argmax(self.density)), self.density.shape))
        rows, cols = self.density.shape
        return {"row": row, "col": col, "label": cell_label(row, col, rows, cols),
                "count": round(float(self.counts[row, col]), 1), "density": round(float(self.density[row, col]), 3)}

    def to_dict(self) -> dict:
        rows, cols = self.counts.shape
        return {"rows": rows, "cols": cols, "counts": np.round(self.counts, 1).tolist(),
                "density": np.round(self.density, 3).tolist(), "hotspot": self.hotspot()}
