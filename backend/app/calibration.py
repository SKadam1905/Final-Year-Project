"""Camera calibration: convert a density map into persons per square metre (plan §4.5).

A homography H maps image pixels on the ground plane to ground coordinates in metres. It
is estimated from >= 4 floor markers measured on site. Each density-map cell covers a
different ground area (far cells cover more), so the zone density is

    (heads in the zone's cells) / (ground area of the same cells)

A homography only describes the ground plane: people on steps or risers add error, which
belongs in the thesis limitations.

tools/calibrate_zone.py makes the markers from one measured rectangle: its four corners clicked
in the camera view, its width and length measured on the ground (Google Maps, a tape).
Pixel coordinates refer to the frame size they were clicked on (ZoneConfig.frame_size); a
camera whose frames arrive at another size (e.g. downscaled to INFER_MAX_WIDTH) is rescaled.
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


def rectangle_ground_points(width_m: float, length_m: float) -> tuple[Point, ...]:
    """Ground positions in metres of a measured rectangle's corners A, B, C, D, in the order they
    are clicked going round it: A at the origin, A->B along the width, B->C along the length."""
    if not (width_m > 0 and length_m > 0):
        raise ValueError("the rectangle's width and length must be positive")
    return (0.0, 0.0), (float(width_m), 0.0), (float(width_m), float(length_m)), (0.0, float(length_m))


def polygon_area(points: Sequence[Point] | np.ndarray) -> float:
    """Area of a simple polygon (shoelace formula), in the square of its coordinates' unit."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))


def ground_polygon(H: np.ndarray, polygon: Sequence[Point]) -> np.ndarray:
    """Ground positions (metres) of an image polygon's vertices.

    A homography maps straight lines to straight lines, so the ground polygon has these vertices
    and straight edges. Raises if a vertex is on or above the horizon: it has no ground position.
    """
    ground, w = project(H, np.asarray(polygon, np.float64))
    if not (w > 1e-12).all() or not np.isfinite(ground).all():
        raise ValueError("part of the polygon is on or above the horizon: outline only ground below it")
    return ground


def ground_area_m2(H: np.ndarray, polygon: Sequence[Point]) -> float:
    """True ground area in m² of an image polygon on the calibrated ground plane."""
    return polygon_area(ground_polygon(H, polygon))


def _cross(o, a, b) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def is_convex(points: Sequence[Point]) -> bool:
    """A strictly convex polygon with its vertices in order (either direction). The image of a
    rectangle on the ground is one, so this catches corners clicked out of order."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    n = len(p)
    if n < 3:
        return False
    turns = [_cross(p[i], p[(i + 1) % n], p[(i + 2) % n]) for i in range(n)]
    if not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
        return False
    # Turns of one sign can still wind round twice (a star); a convex polygon turns exactly once.
    edges = np.roll(p, -1, axis=0) - p
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    turning = (np.roll(angles, -1) - angles + np.pi) % (2 * np.pi) - np.pi
    return bool(abs(abs(turning.sum()) - 2 * np.pi) < 1e-6)


def is_simple_polygon(points: Sequence[Point]) -> bool:
    """At least 3 distinct vertices and no two edges crossing: an outline that does not cross
    itself, so its area is well defined."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    n = len(p)
    if n < 3 or len({(float(x), float(y)) for x, y in p}) < n or polygon_area(p) <= 0:
        return False

    def crosses(a, b, c, d) -> bool:
        d1, d2, d3, d4 = _cross(c, d, a), _cross(c, d, b), _cross(a, b, c), _cross(a, b, d)
        if 0 in (d1, d2, d3, d4):  # touching or collinear: check the overlap directly
            return any(t == 0 and min(e[0], f[0]) <= q[0] <= max(e[0], f[0]) and
                       min(e[1], f[1]) <= q[1] <= max(e[1], f[1])
                       for t, q, e, f in ((d1, a, c, d), (d2, b, c, d), (d3, c, a, b), (d4, d, a, b)))
        return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)

    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # neighbouring edges share a vertex
            if crosses(p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]):
                return False
    return True


def scale_points(points: Sequence[Point], from_size: tuple[int, int], to_size: tuple[int, int]) -> tuple[Point, ...]:
    """Pixel points of a from_size (w, h) frame, in the pixels of the same view at to_size."""
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    return tuple((float(x) * sx, float(y) * sy) for x, y in points)


def rescale_homography(H: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray:
    """The homography for pixels of a to_size frame, given H for the same view at from_size."""
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    return H @ np.diag([1.0 / sx, 1.0 / sy, 1.0])


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

    With a homography the area comes from calibration: each density-map cell in the ROI counts
    with its own ground area, so heads and area cover exactly the same cells. Otherwise it is the
    configured `area_m2` of the ROI (a rough on-site estimate).

    `frame_size` is the (w, h) frame the ROI and homography pixels refer to; frames of another
    size are handled by rescaling both. None: they are in the pixels of whatever frame arrives.
    """

    def __init__(
        self,
        roi: Sequence[Point] | None = None,
        homography: np.ndarray | None = None,
        area_m2: float | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        if homography is None and not area_m2:
            raise ValueError("need a homography or a positive area_m2")
        self.roi = roi
        self.homography = homography
        self.area_m2 = area_m2
        self.frame_size = tuple(frame_size) if frame_size else None
        self._key: tuple | None = None
        self._mask: np.ndarray | None = None
        self._areas: np.ndarray | None = None

    @classmethod
    def from_zone(cls, zone: ZoneConfig) -> ZoneGeometry:
        """The calibrated area when the zone has a calibration, else its estimated area_m2."""
        return cls(zone.roi, zone_homography(zone), zone.area_m2, zone.frame_size)

    def _rescaled(self, frame_size: tuple[int, int]) -> bool:
        return self.frame_size is not None and tuple(frame_size) != self.frame_size

    def roi_at(self, frame_size: tuple[int, int]) -> tuple[Point, ...] | None:
        """The ROI in the pixels of a (w, h) frame."""
        if self.roi is None or not self._rescaled(frame_size):
            return self.roi
        return scale_points(self.roi, self.frame_size, frame_size)

    def homography_at(self, frame_size: tuple[int, int]) -> np.ndarray | None:
        """The homography for the pixels of a (w, h) frame."""
        if self.homography is None or not self._rescaled(frame_size):
            return self.homography
        return rescale_homography(self.homography, self.frame_size, frame_size)

    def _prepare(self, frame_size: tuple[int, int], grid_shape: tuple[int, int]) -> None:
        key = (tuple(frame_size), tuple(grid_shape))
        if key == self._key:
            return
        self._mask = roi_mask(self.roi_at(frame_size), frame_size, grid_shape)
        if self.homography is not None:
            self._areas = cell_ground_areas(self.homography_at(frame_size), frame_size, grid_shape)
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

    def in_zone(self, points: np.ndarray, frame_size: tuple[int, int] | None = None) -> np.ndarray:
        """Which image points (N, 2) of a (w, h) frame, e.g. detected people's feet, lie inside the ROI
        (all without one). Without frame_size the points are taken to be in the ROI's own pixels."""
        points = np.asarray(points, np.float64).reshape(-1, 2)
        roi = self.roi if frame_size is None else self.roi_at(frame_size)
        return np.ones(len(points), bool) if roi is None else points_in_polygon(points, roi)

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
            pts = pts[self.in_zone(pts, frame_size)]
            w, h = frame_size
            pr = np.clip((pts[:, 1] * rows / h).astype(int), 0, rows - 1)
            pc = np.clip((pts[:, 0] * cols / w).astype(int), 0, cols - 1)
            counts = np.bincount(pr * cols + pc, minlength=n).astype(np.float64)
        counts = np.clip(counts, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_m2 = np.where(areas > 0, counts / np.where(areas > 0, areas, 1.0), 0.0)
        return ZoneGrid(counts.reshape(rows, cols), per_m2.reshape(rows, cols))


def zone_homography(zone: ZoneConfig) -> np.ndarray | None:
    """A zone's image -> ground homography: the stored one, else computed from its markers; None if
    the zone is not calibrated."""
    cal = zone.calibration
    if cal is None:
        return None
    if cal.homography is not None:
        return np.asarray(cal.homography, np.float64).reshape(3, 3)
    return compute_homography(cal.image_points, cal.ground_points)


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
