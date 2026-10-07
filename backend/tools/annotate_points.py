"""Click-to-annotate head points: an offline alternative to CVAT.

    python -m tools.annotate_points ../data/india_dataset/frames/pandharpur_01

Mouse:  left click = add head       right click = delete nearest head
        wheel = zoom at cursor      middle-button drag = pan
Keys:   n / space / right arrow = next frame     p / left arrow = previous frame
        u = undo    h = hide/show points    0 = reset zoom    s = save (marks an empty frame as done)
        q / Esc = quit
Every change is saved to data/india_dataset/annotations/<video>/<frame>.json. Existing
files are loaded and the tool opens at the first unannotated frame, so you can stop and
resume at any time. Click the centre of each visible head.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import DATA_DIR
from app.image_io import read_image
from ml.annotations import PointAnnotation, load_points_json, save_points_json

WINDOW = "annotate heads"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LEFT_ARROW, RIGHT_ARROW = 2424832, 2555904  # cv2.waitKeyEx codes on Windows


@dataclass
class View:
    """Maps window pixels to image pixels: image = offset + window / scale."""

    scale: float
    ox: float = 0.0
    oy: float = 0.0

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        return self.ox + x / self.scale, self.oy + y / self.scale

    def to_window(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.ox) * self.scale, (y - self.oy) * self.scale

    def zoom_at(self, factor: float, x: float, y: float, min_scale: float, max_scale: float) -> None:
        """Zoom keeping the image point under window position (x, y) fixed."""
        ix, iy = self.to_image(x, y)
        self.scale = float(np.clip(self.scale * factor, min_scale, max_scale))
        self.ox, self.oy = ix - x / self.scale, iy - y / self.scale

    def clamp(self, image_w: int, image_h: int, win_w: int, win_h: int) -> None:
        self.ox = float(np.clip(self.ox, 0, max(0.0, image_w - win_w / self.scale)))
        self.oy = float(np.clip(self.oy, 0, max(0.0, image_h - win_h / self.scale)))


def nearest_point(points: list[tuple[float, float]], x: float, y: float, max_dist: float) -> int | None:
    if not points:
        return None
    dist = np.hypot(*(np.asarray(points) - [x, y]).T)
    i = int(dist.argmin())
    return i if dist[i] <= max_dist else None


class Annotator:
    def __init__(self, frames: list[Path], out_dir: Path, video: str, window: tuple[int, int] = (1600, 900)):
        self.frames = frames
        self.out_dir = out_dir
        self.video = video
        self.win_w, self.win_h = window
        self.show_points = True
        self.needs_render = True
        self._pan: tuple[int, int, float, float] | None = None
        first_open = next((i for i, f in enumerate(frames) if not self._ann_path(f).exists()), 0)
        self._load(first_open)

    def _ann_path(self, frame: Path) -> Path:
        return self.out_dir / f"{frame.stem}.json"

    def _load(self, index: int) -> None:
        self.index = index
        self.image = read_image(self.frames[index])
        h, w = self.image.shape[:2]
        self.fit_scale = min(self.win_w / w, self.win_h / h)
        self.view = View(self.fit_scale)
        path = self._ann_path(self.frames[index])
        self.points = [tuple(p) for p in load_points_json(path).points.tolist()] if path.exists() else []
        self.history: list[tuple[str, int, tuple[float, float]]] = []
        self.needs_render = True

    def save(self) -> None:
        h, w = self.image.shape[:2]
        frame = self.frames[self.index]
        points = np.asarray(self.points, np.float64).reshape(-1, 2)
        save_points_json(self._ann_path(frame), PointAnnotation(frame.name, self.video, w, h, points))

    def goto(self, index: int) -> None:
        index = min(max(index, 0), len(self.frames) - 1)
        if index != self.index:
            self._load(index)

    def undo(self) -> None:
        if not self.history:
            return
        op, i, point = self.history.pop()
        if op == "add":
            self.points.pop(i)
        else:
            self.points.insert(i, point)
        self.save()

    def _clamp_view(self) -> None:
        h, w = self.image.shape[:2]
        self.view.clamp(w, h, self.win_w, self.win_h)

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        v = self.view
        h, w = self.image.shape[:2]
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = v.to_image(x + 0.5, y + 0.5)  # centre of the clicked window pixel
            if 0 <= ix < w and 0 <= iy < h:
                self.points.append((ix, iy))
                self.history.append(("add", len(self.points) - 1, (ix, iy)))
                self.save()
        elif event == cv2.EVENT_RBUTTONDOWN:
            i = nearest_point(self.points, *v.to_image(x + 0.5, y + 0.5), max_dist=12 / v.scale)
            if i is not None:
                self.history.append(("del", i, self.points.pop(i)))
                self.save()
        elif event == cv2.EVENT_MOUSEWHEEL:
            v.zoom_at(1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8, x, y, self.fit_scale, self.fit_scale * 8)
            self._clamp_view()
        elif event == cv2.EVENT_MBUTTONDOWN:
            self._pan = (x, y, v.ox, v.oy)
        elif event == cv2.EVENT_MBUTTONUP:
            self._pan = None
        elif event == cv2.EVENT_MOUSEMOVE and self._pan and flags & cv2.EVENT_FLAG_MBUTTON:
            sx, sy, ox, oy = self._pan
            v.ox, v.oy = ox - (x - sx) / v.scale, oy - (y - sy) / v.scale
            self._clamp_view()
        else:
            return
        self.needs_render = True

    def render(self) -> None:
        h, w = self.image.shape[:2]
        v = self.view
        size = (min(self.win_w, round(w * self.fit_scale)), min(self.win_h, round(h * self.fit_scale)))
        affine = np.float32([[v.scale, 0, -v.ox * v.scale], [0, v.scale, -v.oy * v.scale]])
        interp = cv2.INTER_NEAREST if v.scale >= 3 else cv2.INTER_LINEAR
        canvas = cv2.warpAffine(self.image, affine, size, flags=interp)
        if self.show_points:
            for x, y in self.points:
                wx, wy = v.to_window(x, y)
                if 0 <= wx < size[0] and 0 <= wy < size[1]:
                    cv2.circle(canvas, (int(wx), int(wy)), 4, (0, 0, 0), -1)
                    cv2.circle(canvas, (int(wx), int(wy)), 3, (0, 0, 255), -1)
        text = (f"{self.video}/{self.frames[self.index].name}  [{self.index + 1}/{len(self.frames)}]  "
                f"heads: {len(self.points)}  zoom {v.scale / self.fit_scale:.1f}x")
        cv2.rectangle(canvas, (0, 0), (size[0], 28), (0, 0, 0), -1)
        cv2.putText(canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, canvas)

    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            if self.needs_render:
                self.render()
                self.needs_render = False
            key = cv2.waitKeyEx(20)
            if key == -1:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed with the X button
                continue
            c = chr(key).lower() if 0 <= key < 256 else ""
            if c == "q" or key == 27:
                break
            if c in ("n", " ") or key == RIGHT_ARROW:
                self.goto(self.index + 1)
            elif c == "p" or key == LEFT_ARROW:
                self.goto(self.index - 1)
            elif c == "u":
                self.undo()
            elif c == "h":
                self.show_points = not self.show_points
            elif c == "0":
                self.view = View(self.fit_scale)
            elif c == "s":
                self.save()
            self.needs_render = True
        cv2.destroyAllWindows()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames_dir", type=Path, help="folder of frames for one video")
    p.add_argument("--out", type=Path, help="default: data/india_dataset/annotations/<video>")
    p.add_argument("--window", default="1600x900", help="maximum window size, WxH")
    args = p.parse_args(argv)
    frames = sorted(f for f in args.frames_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not frames:
        raise SystemExit(f"no images in {args.frames_dir}")
    video = args.frames_dir.name
    out = args.out or DATA_DIR / "india_dataset" / "annotations" / video
    win_w, win_h = (int(v) for v in args.window.lower().split("x"))
    print(__doc__)
    Annotator(frames, out, video, (win_w, win_h)).run()
    done = sum((out / f"{f.stem}.json").exists() for f in frames)
    print(f"{done}/{len(frames)} frames annotated -> {out}")


if __name__ == "__main__":
    main()
