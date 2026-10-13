r"""Calibrate a zone's ground area from its camera view: a measured rectangle -> homography -> true m².

    cd backend
    ..\venv\Scripts\python.exe -m tools.calibrate_zone --zone RAMKUND_NORTH --method google_maps ^
        --measured-by shubham --note "paved platform west of the kund, corners at the 4 lamp posts"
    ..\venv\Scripts\python.exe -m tools.calibrate_zone ..\data\calibration\ramkund_cam1.mp4 --zone RAMKUND_NORTH ^
        --method tape --time 42

The view is a video (the frame at --time seconds; , and . step 1 s), an image, or by default the zone's
own clip. Calibrate a fixed camera: a calibration holds only for the view it was made on, so a news
camera that pans or zooms cannot be calibrated once and for all.

1. Click the 4 corners of a rectangle on the ground, going round it: A, B, C, D. A->B is its width and
   B->C its length. Pick a large rectangle near the middle of the zone, with corners you can place
   exactly (paving joints, the edges of a platform or a step); see results/CALIBRATION_GUIDE.md.
2. Type its real width A->B in metres, Enter, then its length B->C, Enter (or give --width/--length).
   A ground grid is then drawn over the view: its lines must lie flat on the ground.
3. Click the corners of the zone's ground polygon, the walkable ground whose people are counted, and
   press Enter to close it. Only ground below the horizon; its area in m² is shown as you click.
4. Check the summary line and press w to save to config/zones.json.

Keys:   right click / u = undo the last point (u in the summary = edit the polygon)   r = start again
        c = contrast + sharpening (display only)   wheel = zoom (up to 6x)   middle drag = pan
        0 = whole frame   , / . = 1 s back / forward (video)   q / Esc = quit (unsaved work is lost)

Saved for the zone: roi (the polygon, camera pixels), area_m2 (its calibrated area), frame_size (the
pixels' frame) and calibration {image_points, ground_points, homography, source}: source records the
method (--method), the rectangle (--note, width, length), who measured it and when, the view, and the
area's spread for 1-pixel click errors. The first calibration keeps the previous area_m2 and roi as
area_m2_estimated and roi_estimated for comparison (python -m tools.calibration_report). The view with
the overlay goes to data/calibration/<zone>.jpg (git-ignored: it can show faces).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from app.calibration import (compute_homography, ground_area_m2, ground_polygon, is_convex, is_simple_polygon,
                             project, rectangle_ground_points)
from app.config import DATA_DIR, REPO_ROOT, load_zones
from app.image_io import read_image
from tools.annotate_points import MAX_ZOOM, ZOOM_STEP, View, enhance, wheel_delta

WINDOW = "calibrate zone"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ENTER, BACKSPACE, ESC = (10, 13), 8, 27
BAR = 28  # status bar line height
TOP = 2 * BAR  # the status lines sit above the picture
METHODS = ("google_maps", "tape", "survey", "other")
ZONE_KEYS = ("id", "name", "source", "graph_node", "fixed_camera", "roi", "area_m2", "area_m2_estimated",
             "roi_estimated", "frame_size", "calibration")
CORNERS = "ABCD"
CLICK_SD_PX = 1.0  # click error assumed for the area's spread

Point = tuple[float, float]


# calibration -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    image_points: tuple[Point, ...]  # rectangle corners A, B, C, D in pixels (0.1 px)
    ground_points: tuple[Point, ...]  # their ground positions in metres
    homography: np.ndarray  # pixels -> metres
    roi: tuple[Point, ...]  # the zone's ground polygon in pixels (0.1 px)
    area_m2: float  # the polygon's true ground area
    rectangle_m2: float
    reach: float  # farthest polygon corner from the rectangle's centre, in rectangle half-diagonals
    click_sd_pct: float  # spread (1 sd) of area_m2 for CLICK_SD_PX click errors, in % of area_m2


def _round(points: Sequence[Point]) -> tuple[Point, ...]:
    return tuple((round(float(x), 1), round(float(y), 1)) for x, y in points)


def click_spread(corners: Sequence[Point], ground: Sequence[Point], polygon: Sequence[Point],
                 sd_px: float = CLICK_SD_PX, trials: int = 300, seed: int = 0) -> float:
    """Relative spread (1 sd, %) of the zone area when every clicked point (the rectangle's corners and
    the polygon's) is off by a random sd_px pixels: how much the area depends on clicking precisely."""
    rng = np.random.default_rng(seed)
    corners, polygon = np.asarray(corners, np.float64), np.asarray(polygon, np.float64)
    areas = []
    for _ in range(trials):
        try:
            H = compute_homography(corners + rng.normal(0, sd_px, corners.shape), ground)
            areas.append(ground_area_m2(H, polygon + rng.normal(0, sd_px, polygon.shape)))
        except ValueError:  # a jittered point fell beyond the horizon: leave that trial out
            continue
    if len(areas) < trials // 2:
        return float("inf")
    return 100.0 * float(np.std(areas)) / float(np.mean(areas))


def calibrate(corners: Sequence[Point], width_m: float, length_m: float, polygon: Sequence[Point]) -> Result:
    """Homography from the rectangle's clicked corners A, B, C, D (going round it; A->B = width, B->C =
    length) and the zone polygon's true ground area. Points are rounded to 0.1 px first, so the saved
    points reproduce the saved homography and area exactly."""
    corners, polygon = _round(corners), _round(polygon)
    if len(corners) != 4:
        raise ValueError("click the rectangle's 4 corners")
    if not is_convex(corners):
        raise ValueError("A, B, C, D must go round the rectangle in order")
    if len(polygon) < 3 or not is_simple_polygon(polygon):
        raise ValueError("the zone polygon needs 3+ corners and must not cross itself")
    ground = rectangle_ground_points(width_m, length_m)
    H = compute_homography(corners, ground)
    area = ground_area_m2(H, polygon)
    centre = np.array([width_m / 2, length_m / 2])
    reach = float(np.hypot(*(ground_polygon(H, polygon) - centre).T).max() / np.hypot(width_m / 2, length_m / 2))
    return Result(corners, ground, H, polygon, area, width_m * length_m, reach, click_spread(corners, ground, polygon))


# config/zones.json ------------------------------------------------------------------------------

def apply_calibration(data: dict, zone_id: str, result: Result, frame_size: tuple[int, int], source: dict) -> dict:
    """zones.json content with the zone calibrated: roi, area_m2, frame_size and calibration replaced.
    The first time, the previous area_m2 and roi are kept as area_m2_estimated and roi_estimated; a
    recalibration leaves those alone, so they always hold the original estimate."""
    zones = data["zones"]
    index = next((i for i, z in enumerate(zones) if z["id"] == zone_id), None)
    if index is None:
        raise ValueError(f"unknown zone {zone_id}; zones: {', '.join(z['id'] for z in zones)}")
    zone = dict(zones[index])
    if "area_m2_estimated" not in zone:
        zone["area_m2_estimated"] = zone.get("area_m2")
        zone["roi_estimated"] = zone.get("roi")
    zone["roi"] = [list(p) for p in result.roi]
    zone["area_m2"] = round(result.area_m2, 1)
    zone["frame_size"] = [int(frame_size[0]), int(frame_size[1])]
    zone["calibration"] = {
        "image_points": [list(p) for p in result.image_points],
        "ground_points": [list(p) for p in result.ground_points],
        "homography": [[float(v) for v in row] for row in result.homography],
        "source": {k: v for k, v in source.items() if v not in (None, "")},
    }
    zone = {k: zone[k] for k in ZONE_KEYS if k in zone} | {k: v for k, v in zone.items() if k not in ZONE_KEYS}
    return {**data, "zones": [*zones[:index], zone, *zones[index + 1:]]}


def _is_numbers(value) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)


def dumps_zones(data: dict) -> str:
    """JSON with 2-space indents, but every list of numbers or of number pairs on one line."""
    compact: dict[str, str] = {}

    def mark(value):
        if isinstance(value, dict):
            return {k: mark(v) for k, v in value.items()}
        if isinstance(value, list):
            if _is_numbers(value) or (value and all(_is_numbers(v) for v in value)):
                key = f"@@{len(compact)}@@"
                compact[key] = json.dumps(value)
                return key
            return [mark(v) for v in value]
        return value

    text = json.dumps(mark(data), indent=2, ensure_ascii=False)
    for key, value in compact.items():
        text = text.replace(f'"{key}"', value)
    return text + "\n"


def save_zones(path: Path, data: dict) -> None:
    """Write zones.json only if the result loads (load_zones checks the homography and area), via a
    temporary file, so a failed save never leaves a broken config."""
    text = dumps_zones(data)
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        load_zones(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def display_path(path: Path) -> str:
    """Repo-relative path for the record (no user folders); just the name outside the repo."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name


# drawing helpers --------------------------------------------------------------------------------

def nice_step(extent_m: float, max_lines: int = 12) -> float:
    """A round grid spacing (0.5, 1, 2, 5, 10 ... m) giving at most max_lines lines over extent_m."""
    for step in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500):
        if extent_m / step <= max_lines:
            return float(step)
    return 1000.0


def ground_grid(H: np.ndarray, bounds: tuple[float, float, float, float], step: float,
                samples: int = 60) -> list[np.ndarray]:
    """Image polylines of ground lines every `step` m over bounds (x0, y0, x1, y1) in metres; the parts
    beyond the horizon are left out."""
    Hinv = np.linalg.inv(H)
    x0, y0, x1, y1 = bounds
    lines = []
    for fixed, other, axis in ((np.arange(np.floor(x0 / step), np.ceil(x1 / step) + 1) * step, (y0, y1), 0),
                               (np.arange(np.floor(y0 / step), np.ceil(y1 / step) + 1) * step, (x0, x1), 1)):
        t = np.linspace(*other, samples)
        for value in fixed:
            ground = np.column_stack([np.full(samples, value), t] if axis == 0 else [t, np.full(samples, value)])
            homog = np.column_stack([ground, np.ones(samples)]) @ Hinv.T
            front = homog[:, 2] > 1e-9  # in front of the camera (H is scaled positive there)
            with np.errstate(divide="ignore", invalid="ignore"):
                pixels = homog[:, :2] / homog[:, 2:]
            ok = front & (np.abs(pixels) < 1e5).all(axis=1)
            run: list[np.ndarray] = []
            for good, p in zip(ok, pixels):
                if good:
                    run.append(p)
                elif len(run) > 1:
                    lines.append(np.array(run))
                    run = []
                else:
                    run = []
            if len(run) > 1:
                lines.append(np.array(run))
    return lines


# the tool ---------------------------------------------------------------------------------------

class Calibrator:
    """Rectangle corners -> width and length -> zone polygon -> summary; on_save(result) stores it."""

    def __init__(self, image: np.ndarray, zone_id: str, on_save: Callable[[Result], str],
                 estimate: float | None = None, window: tuple[int, int] = (1600, 900),
                 width_m: float | None = None, length_m: float | None = None,
                 step_frame: Callable[[float], tuple[np.ndarray, str] | None] | None = None, frame_label: str = ""):
        self.zone_id, self.on_save, self.estimate = zone_id, on_save, estimate
        self.win_w, self.win_h = window
        self.preset = (width_m, length_m) if width_m and length_m else None
        self.step_frame, self.frame_label = step_frame, frame_label
        self.enhanced, self.needs_render = False, True
        self._pan: tuple[int, int, float, float] | None = None
        self.set_image(image)
        self.restart()

    def set_image(self, image: np.ndarray) -> None:
        self.image = image
        self._enhanced_image: np.ndarray | None = None
        h, w = image.shape[:2]
        self.fit_scale = min(self.win_w / w, (self.win_h - TOP) / h)
        self.size = (round(w * self.fit_scale), round(h * self.fit_scale))
        self.view = View(self.fit_scale)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    def restart(self) -> None:
        self.corners: list[Point] = []
        self.polygon: list[Point] = []
        self.width_m = self.length_m = None
        self.H: np.ndarray | None = None
        self.result: Result | None = None
        self.entry, self.phase, self.saved = "", "corners", False
        self.message = "click corner A of the measured rectangle"

    # state changes --------------------------------------------------------------------------------

    def add_point(self, x: float, y: float) -> None:
        w, h = self.frame_size
        if not (0 <= x < w and 0 <= y < h):
            return
        if self.phase == "corners":
            if len(self.corners) == 4:
                self.message = "A, B, C, D must go round the rectangle in order: u = undo D, r = start again"
                return
            self.corners.append((x, y))
            if len(self.corners) < 4:
                self.message = f"click corner {CORNERS[len(self.corners)]}" + (
                    " (A->B is the width)" if len(self.corners) == 1 else
                    " (B->C is the length)" if len(self.corners) == 2 else "")
            elif not is_convex(self.corners):
                self.message = "A, B, C, D must go round the rectangle in order: u = undo D, r = start again"
            elif self.preset:
                self.width_m, self.length_m = self.preset
                self.fit_homography()
            else:
                self.phase, self.entry = "width", ""
                self.message = "type the rectangle's width A->B in metres, then Enter"
        elif self.phase == "polygon":
            if project(self.H, np.array([[x, y]]))[1][0] <= 1e-9:
                self.message = "that point is above the horizon: the zone must be ground below it"
                return
            self.polygon.append((x, y))
            self.message = self.polygon_message()

    def undo(self) -> None:
        if self.phase == "corners" and self.corners:
            self.corners.pop()
            self.message = f"click corner {CORNERS[len(self.corners)]}"
        elif self.phase == "width":
            self.phase = "corners"
            self.corners.pop()
            self.message = "click corner D"
        elif self.phase == "length":
            self.phase, self.entry = "width", f"{self.width_m:g}"
            self.message = "type the rectangle's width A->B in metres, then Enter"
        elif self.phase == "polygon" and self.polygon:
            self.polygon.pop()
            self.message = self.polygon_message()
        elif self.phase == "polygon" and self.preset:
            self.phase, self.H = "corners", None
            self.corners.pop()
            self.message = "click corner D"
        elif self.phase == "polygon":
            self.phase, self.entry, self.H = "length", f"{self.length_m:g}", None
            self.message = "type the rectangle's length B->C in metres, then Enter"
        elif self.phase == "review":
            self.phase, self.result = "polygon", None
            self.message = self.polygon_message()

    def commit_number(self) -> None:
        try:
            value = float(self.entry)
        except ValueError:
            value = 0.0
        if not 0.1 <= value <= 2000:
            self.message = "type a length between 0.1 and 2000 m"
            return
        self.entry = ""
        if self.phase == "width":
            self.width_m, self.phase = value, "length"
            self.message = "type the rectangle's length B->C in metres, then Enter"
        else:
            self.length_m = value
            self.fit_homography()

    def fit_homography(self) -> None:
        try:
            self.H = compute_homography(self.corners, rectangle_ground_points(self.width_m, self.length_m))
        except ValueError as exc:
            self.message = f"{exc}: u = undo, r = start again"
            return
        self.phase = "polygon"
        self.message = ("the grid must lie flat on the ground (else u / r). Now click the zone's ground polygon, "
                        "Enter = close it")

    def close_polygon(self) -> None:
        try:
            self.result = calibrate(self.corners, self.width_m, self.length_m, self.polygon)
        except ValueError as exc:
            self.message = str(exc)
            return
        self.phase = "review"
        self.message = self.summary()

    def polygon_area(self) -> float | None:
        if self.H is None or len(self.polygon) < 3:
            return None
        try:
            return ground_area_m2(self.H, self.polygon)
        except ValueError:
            return None

    def polygon_message(self) -> str:
        area = self.polygon_area()
        return (f"zone polygon: {len(self.polygon)} corners" + (f", {area:,.0f} m2" if area else "") +
                "   Enter = close it   u / right click = undo")

    def summary(self) -> str:
        r = self.result
        text = (f"rectangle {self.width_m:g} x {self.length_m:g} m; zone {r.area_m2:,.1f} m2 "
                f"(+/- {r.click_sd_pct:.1f}% for {CLICK_SD_PX:g} px click errors)")
        if self.estimate:
            text += f"; estimate was {self.estimate:,.0f} m2 ({self.estimate / r.area_m2:.2f}x)"
        if r.reach > 3:
            text += f"; zone reaches {r.reach:.1f}x the rectangle's size away: errors grow there"
        return text + "   w = save   u = edit polygon"

    def save(self) -> None:
        if self.phase != "review":
            self.message = "finish the polygon first (Enter closes it)"
            return
        try:
            self.message = self.on_save(self.result)
            self.saved = True
        except (OSError, ValueError) as exc:
            self.message = f"not saved: {exc}"

    # input ----------------------------------------------------------------------------------------

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param=None) -> None:
        y -= TOP  # window pixels -> picture pixels
        if y < 0 and event != cv2.EVENT_MOUSEWHEEL:
            return
        v = self.view
        h, w = self.image.shape[:2]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(*v.to_image(x + 0.5, y + 0.5))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo()
        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = ZOOM_STEP if wheel_delta(flags) > 0 else 1 / ZOOM_STEP
            v.zoom_at(factor, x, y, self.fit_scale, self.fit_scale * MAX_ZOOM)
            v.clamp(w, h, *self.size)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self._pan = (x, y, v.ox, v.oy)
        elif event == cv2.EVENT_MBUTTONUP:
            self._pan = None
        elif event == cv2.EVENT_MOUSEMOVE and self._pan and flags & cv2.EVENT_FLAG_MBUTTON:
            sx, sy, ox, oy = self._pan
            v.ox, v.oy = ox - (x - sx) / v.scale, oy - (y - sy) / v.scale
            v.clamp(w, h, *self.size)
        else:
            return
        self.needs_render = True

    def handle_key(self, key: int) -> bool:
        """Act on a key from cv2.waitKeyEx; False means quit."""
        c = chr(key).lower() if 0 <= key < 256 else ""
        self.needs_render = True
        if self.phase in ("width", "length"):
            c = "." if c == "," else c  # a decimal comma works too
            if c.isdigit() or (c == "." and "." not in self.entry):
                if len(self.entry) < 7:
                    self.entry += c
                return True
            if key == BACKSPACE:
                self.entry = self.entry[:-1]
                return True
            if key in ENTER:
                self.commit_number()
                return True
        if c == "q" or key == ESC:
            return False
        if key in ENTER and self.phase == "polygon":
            self.close_polygon()
        elif c == "u" or (key == BACKSPACE and self.phase in ("corners", "polygon")):
            self.undo()
        elif c == "r":
            self.restart()
        elif c == "w":
            self.save()
        elif c == "c":
            self.enhanced = not self.enhanced
            self._enhanced_image = None
        elif c == "0":
            self.view = View(self.fit_scale)
        elif c in (",", ".") and self.step_frame is not None:
            step = self.step_frame(-1.0 if c == "," else 1.0)
            if step is None:
                self.message = "no frame there"
            else:
                image, self.frame_label = step
                view = self.view
                self.set_image(image)
                self.view = view  # same camera view: keep the zoom and the points
        return True

    # drawing --------------------------------------------------------------------------------------

    def _display_image(self) -> np.ndarray:
        if not self.enhanced:
            return self.image
        if self._enhanced_image is None:
            self._enhanced_image = enhance(self.image)
        return self._enhanced_image

    def grid_bounds(self) -> tuple[float, float, float, float]:
        x = [0.0, float(self.width_m)]
        y = [0.0, float(self.length_m)]
        if self.polygon:
            ground, w = project(self.H, np.asarray(self.polygon))
            ground = ground[w > 1e-9]
            x += ground[:, 0].tolist()
            y += ground[:, 1].tolist()
        return min(x), min(y), max(x), max(y)

    def draw(self, view: View, size: tuple[int, int]) -> np.ndarray:
        affine = np.float32([[view.scale, 0, -view.ox * view.scale], [0, view.scale, -view.oy * view.scale]])
        canvas = cv2.warpAffine(self._display_image(), affine, size, flags=cv2.INTER_CUBIC)

        def win(points) -> np.ndarray:
            return np.array([view.to_window(x, y) for x, y in points]).round().astype(np.int32)

        def label(text: str, at, color=(255, 255, 255)) -> None:
            at = tuple(int(v) for v in at)
            cv2.putText(canvas, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        if self.H is not None:
            x0, y0, x1, y1 = self.grid_bounds()
            step = nice_step(max(x1 - x0, y1 - y0))
            for line in ground_grid(self.H, (x0 - step, y0 - step, x1 + step, y1 + step), step):
                cv2.polylines(canvas, [win(line)], False, (80, 200, 80), 1, cv2.LINE_AA)
            label(f"grid: {step:g} m", (8, size[1] - 10), (80, 200, 80))
        if self.polygon:
            pts = win(self.polygon)
            if self.phase == "review":
                shade = canvas.copy()
                cv2.fillPoly(shade, [pts], (0, 160, 255))
                canvas = cv2.addWeighted(shade, 0.25, canvas, 0.75, 0)
            cv2.polylines(canvas, [pts], self.phase == "review", (0, 160, 255), 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(canvas, tuple(int(v) for v in p), 4, (0, 160, 255), -1)
            area = self.result.area_m2 if self.result else self.polygon_area()
            if area:
                label(f"{area:,.0f} m2", pts.mean(axis=0), (0, 200, 255))
        if self.corners:
            pts = win(self.corners)
            cv2.polylines(canvas, [pts], len(pts) == 4, (255, 255, 0), 2, cv2.LINE_AA)
            for name, p in zip(CORNERS, pts):
                cv2.circle(canvas, tuple(int(v) for v in p), 5, (255, 255, 0), -1)
                label(name, p + (8, -8), (255, 255, 0))
            if self.width_m and len(pts) >= 2:
                label(f"{self.width_m:g} m", (pts[0] + pts[1]) / 2 + (6, 18), (255, 255, 0))
            if self.length_m and len(pts) >= 3:
                label(f"{self.length_m:g} m", (pts[1] + pts[2]) / 2 + (6, 18), (255, 255, 0))
        return canvas

    def status_lines(self) -> tuple[str, str]:
        typing = (f"   {self.phase}: {self.entry}_ m" if self.phase in ("width", "length") else "")
        line = (f"zone {self.zone_id}   {self.frame_label}   step: {self.phase}{typing}   "
                f"zoom {self.view.scale / self.fit_scale:.1f}x{'   enhanced' if self.enhanced else ''}")
        return line + f"   | {self.message}", ("left click = point   right click / u = undo   Enter = next   "
                                               "r = start again   w = save   c contrast   , . = 1 s   q quit")

    def render(self) -> np.ndarray:
        canvas = self.draw(self.view, self.size)
        bars = np.zeros((TOP, self.size[0], 3), np.uint8)
        for i, text in enumerate(self.status_lines()):
            cv2.putText(bars, text, (8, i * BAR + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bars, canvas])

    def snapshot(self) -> np.ndarray:
        """The whole view with the rectangle, ground grid and zone polygon (for data/calibration/)."""
        enhanced, self.enhanced = self.enhanced, False
        image = self.draw(View(self.fit_scale), self.size)
        self.enhanced = enhanced
        return image

    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            if self.needs_render:
                cv2.imshow(WINDOW, self.render())
                self.needs_render = False
            key = cv2.waitKeyEx(20)
            if key == -1:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed with the X button
                continue
            if not self.handle_key(key):
                break
        cv2.destroyAllWindows()


# input frames -----------------------------------------------------------------------------------

class VideoFrames:
    """Frames of a video by time, for picking a moment where the rectangle's corners are visible."""

    def __init__(self, path: Path, time_s: float):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.duration = (self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        self.time_s = time_s

    def read(self, time_s: float) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000)
        ok, frame = self.cap.read()
        if ok:
            self.time_s = max(0.0, time_s)
        return frame if ok else None

    def step(self, seconds: float) -> tuple[np.ndarray, str] | None:
        frame = self.read(self.time_s + seconds)
        return None if frame is None else (frame, self.label())

    def label(self) -> str:
        return f"{self.path.name} at {self.time_s:.1f} s"


def main(argv=None) -> None:
    from app.config import get_settings

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("view", nargs="?", type=Path, help="video or image of the camera view (default: the zone's clip)")
    p.add_argument("--zone", required=True, help="zone id in config/zones.json")
    p.add_argument("--method", required=True, choices=METHODS, help="how the rectangle was measured")
    p.add_argument("--note", help="which rectangle: where it is and what marks its corners")
    p.add_argument("--measured-by", help="who measured the rectangle")
    p.add_argument("--measured-on", help="when it was measured, YYYY-MM-DD")
    p.add_argument("--width", type=float, help="the rectangle's width A->B in metres (else typed in the window)")
    p.add_argument("--length", type=float, help="its length B->C in metres")
    p.add_argument("--time", type=float, default=0.0, help="video only: the frame at this many seconds")
    p.add_argument("--config", type=Path, help="default: ZONES_CONFIG from .env (config/zones.json)")
    p.add_argument("--window", default="1600x900", help="maximum window size, WxH")
    args = p.parse_args(argv)
    if args.measured_on:
        date.fromisoformat(args.measured_on)  # fails early on a typo
    config = args.config or Path(get_settings().zones_config)
    data = json.loads(config.read_text(encoding="utf-8"))
    raw = next((z for z in data["zones"] if z["id"] == args.zone), None)
    if raw is None:
        raise SystemExit(f"unknown zone {args.zone}; zones: {', '.join(z['id'] for z in data['zones'])}")
    view = args.view
    if view is None:
        from app.worker import local_source

        zone = next(z for z in load_zones(config) if z.id == args.zone)
        view = Path(local_source(zone.source))
        if not view.exists():
            raise SystemExit(f"no local clip for {zone.source}: give a video or image of the camera view")

    frames = None
    if view.suffix.lower() in IMAGE_EXTS:
        image, label = read_image(view), view.name
    else:
        frames = VideoFrames(view, args.time)
        image = frames.read(args.time)
        if image is None:
            raise SystemExit(f"no frame at {args.time} s in {view}")
        label = frames.label()
    estimate = raw.get("area_m2_estimated", raw.get("area_m2"))
    snapshots = DATA_DIR / "calibration"

    def on_save(result: Result) -> str:
        frame_label = tool.frame_label
        source = {
            "method": args.method, "rectangle": args.note, "width_m": tool.width_m, "length_m": tool.length_m,
            "measured_by": args.measured_by, "measured_on": args.measured_on,
            "calibrated_on": datetime.now().isoformat(timespec="minutes"),
            "view": display_path(view) + (frame_label[len(view.name):] if frame_label.startswith(view.name) else ""),
            "area_click_sd_pct": round(result.click_sd_pct, 2), "tool": "tools/calibrate_zone.py",
        }
        current = json.loads(config.read_text(encoding="utf-8"))  # someone may have edited it meanwhile
        save_zones(config, apply_calibration(current, args.zone, result, tool.frame_size, source))
        snapshots.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snapshots / f"{args.zone}.jpg"), tool.snapshot())
        print(f"{args.zone}: {result.area_m2:,.1f} m² (+/- {result.click_sd_pct:.1f}% for {CLICK_SD_PX:g} px clicks; "
              f"estimate {estimate} m²) -> {config}; overlay -> {snapshots / (args.zone + '.jpg')}")
        return f"saved to {config.name} ({result.area_m2:,.1f} m2); q = quit"

    win_w, win_h = (int(v) for v in args.window.lower().split("x"))
    print(__doc__)
    tool = Calibrator(image, args.zone, on_save, estimate, (win_w, win_h), args.width, args.length,
                      frames.step if frames else None, label)
    tool.run()
    if not tool.saved:
        print("not saved")


if __name__ == "__main__":
    main()
