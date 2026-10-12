r"""Count heads per grid cell on the Kumbh evaluation frames: a faster ground truth than clicking every head.

    cd backend
    ..\venv\Scripts\python.exe -m tools.count_eval

Each usable frame (not marked unreadable) is shown with its counting region (COUNT_REGIONS in
tools.make_kumbh_eval; otherwise the whole frame) split into a 3x3 grid. The cyan cell is zoomed in:
type how many heads you see in it and press Enter (Enter alone = 0; on a cell already counted it keeps
the count), and the next cell follows. Count
a head in the cell that holds its centre, so a head on a grid line is counted once. Nothing is
pre-filled and no model is loaded: the counts are a person's alone.

Keys:   0-9 = type the count     Backspace = delete a digit     Enter = save the cell, go to the next
        b = previous cell        click = pick a cell             z = cell view / whole frame
        c = contrast + sharpening (display only)                 wheel = zoom (up to 6x), middle drag = pan
        n / p = next / previous frame                            s = mark the frame unreadable (then 1-5)
        r r = redraw the counting region (clears this frame's counts), then drag it
        q / Esc = quit
Every Enter saves to data/eval/annotations/kumbh_eval/counts.csv: the region, each cell's count,
the total and whether all nine cells are done. The tool reopens at the first frame not complete.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from app.config import DATA_DIR
from app.image_io import read_image
from ml.annotations import UNREADABLE, read_frame_status, set_frame_status
from ml.kumbh_eval import CELLS, COUNTS_NAME, cell_rects, count_row, read_counts, write_counts
from tools.annotate_points import MAX_ZOOM, REASONS, ZOOM_STEP, View, enhance, roi_from_drag, wheel_delta
from tools.make_kumbh_eval import COUNT_REGIONS

WINDOW = "count heads"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ENTER, BACKSPACE, ESC = (10, 13), 8, 27
BAR = 28  # status bar line height
TOP = 2 * BAR  # the two status lines sit above the picture, never over it
HELP = ("0-9 + Enter = heads in the cyan cell (Enter alone = 0)   b back   click = pick cell   z whole frame   "
        "c contrast   s unreadable   r r region   n/p frame   q quit")


class CountTool:
    def __init__(self, frames_dir: Path, counts_path: Path, window: tuple[int, int] = (1600, 900),
                 regions: dict[str, tuple] = COUNT_REGIONS):
        self.frames_dir, self.counts_path, self.regions = frames_dir, counts_path, regions
        self.frames_csv = frames_dir / "frames.csv"
        self.all_frames = sorted(f for f in frames_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        with open(self.frames_csv, newline="", encoding="utf-8-sig") as f:
            self.shot_type = {r["frame"]: r.get("shot_type", "") for r in csv.DictReader(f)}
        self.win_w, self.win_h = window
        self.rows = read_counts(counts_path)
        self.enhanced, self.cell_view, self.needs_render = False, True, True
        self._pan: tuple[int, int, float, float] | None = None
        self._drag: tuple[float, float, float, float] | None = None
        usable = self.usable()
        if not usable:
            raise SystemExit("no usable frames: every frame is marked unreadable")
        self.load(next((f for f in usable if self.rows.get(f.name, {}).get("complete") != "yes"), usable[0]))

    # frames ---------------------------------------------------------------------------------------

    def usable(self) -> list[Path]:
        status = read_frame_status(self.frames_csv)
        return [f for f in self.all_frames if status.get(f.name, ("",))[0] != UNREADABLE]

    def load(self, frame: Path) -> None:
        self.frame = frame
        self.image = read_image(frame)
        self._enhanced_image: np.ndarray | None = None
        h, w = self.image.shape[:2]
        self.fit_scale = min(self.win_w / w, (self.win_h - TOP) / h)  # --window is the whole window
        self.size = (round(w * self.fit_scale), round(h * self.fit_scale))  # the picture's part of it
        row = self.rows.get(frame.name)
        if row:
            self.region = tuple(float(row[k]) for k in ("x0", "y0", "x1", "y1"))
            self.cells: list[int | None] = [int(row[c]) if row.get(c, "") != "" else None for c in CELLS]
        else:
            self.region = tuple(float(v) for v in self.regions.get(frame.name, (0, 0, w, h)))
            self.cells = [None] * len(CELLS)
        self.cell = next((k for k, c in enumerate(self.cells) if c is None), 0)
        self.entry, self.message = "", ""
        self.choosing_reason = self.redraw_armed = self.drawing_region = False
        self._drag = None
        self.cell_view = True
        self.focus()

    def step_frame(self, direction: int) -> None:
        position = self.all_frames.index(self.frame)
        usable = set(self.usable())
        candidates = (self.all_frames[position + 1:] if direction > 0 else self.all_frames[:position][::-1])
        target = next((f for f in candidates if f in usable), None)
        if target is None:
            self.message = "this is the last usable frame" if direction > 0 else "this is the first usable frame"
        else:
            self.load(target)

    # counts ---------------------------------------------------------------------------------------

    def save(self) -> bool:
        self.rows[self.frame.name] = count_row(self.frame.name, self.region, self.cells)
        try:
            write_counts(self.counts_path, self.rows)
        except OSError as exc:  # e.g. counts.csv open in Excel, which locks it
            self.message = f"could not save {self.counts_path.name} ({exc.strerror}); close it in other programs"
            return False
        return True

    def commit_entry(self) -> None:
        """Enter: the typed count, or 0 when nothing was typed; a counted cell keeps its count on a bare Enter."""
        if self.entry or self.cells[self.cell] is None:
            self.cells[self.cell] = int(self.entry) if self.entry else 0
        self.entry = ""
        if not self.save():
            return
        if all(c is not None for c in self.cells):
            if self.cell == len(self.cells) - 1:
                self.message = f"frame complete: {sum(self.cells)} heads. n = next frame"
                return
            self.select(self.cell + 1)
        else:
            later = [k for k in range(self.cell + 1, len(self.cells)) if self.cells[k] is None]
            self.select(later[0] if later else next(k for k, c in enumerate(self.cells) if c is None))
        self.message = "saved"

    def select(self, cell: int) -> None:
        self.cell, self.entry, self.cell_view = cell, "", True
        self.focus()

    # view -----------------------------------------------------------------------------------------

    def focus(self) -> None:
        """Cell view: zoom so the current cell (plus a margin) fills the window; else the whole frame."""
        h, w = self.image.shape[:2]
        if not self.cell_view:
            self.view = View(self.fit_scale)
            return
        x0, y0, x1, y1 = cell_rects(self.region)[self.cell]
        scale = min(self.size[0] / ((x1 - x0) * 1.2), self.size[1] / ((y1 - y0) * 1.2))
        scale = float(np.clip(scale, self.fit_scale, self.fit_scale * MAX_ZOOM))
        self.view = View(scale, (x0 + x1) / 2 - self.size[0] / 2 / scale, (y0 + y1) / 2 - self.size[1] / 2 / scale)
        self.view.clamp(w, h, *self.size)

    def cell_at(self, x: float, y: float) -> int | None:
        for k, (x0, y0, x1, y1) in enumerate(cell_rects(self.region)):
            if x0 <= x < x1 and y0 <= y < y1:
                return k
        return None

    # input ----------------------------------------------------------------------------------------

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        y -= TOP  # window pixels -> picture pixels
        if y < 0 and event != cv2.EVENT_MOUSEWHEEL:
            return  # on the status lines
        v = self.view
        h, w = self.image.shape[:2]
        ix, iy = v.to_image(x + 0.5, y + 0.5)
        if self.drawing_region and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_MOUSEMOVE):
            if event == cv2.EVENT_LBUTTONDOWN:
                self._drag = (ix, iy, ix, iy)
            elif self._drag is None or (event == cv2.EVENT_MOUSEMOVE and not flags & cv2.EVENT_FLAG_LBUTTON):
                return
            elif event == cv2.EVENT_MOUSEMOVE:
                self._drag = (*self._drag[:2], ix, iy)
            else:
                region = roi_from_drag(*self._drag[:2], ix, iy, w, h, min_size=24)
                self._drag = None
                if region is None:
                    self.message = "region too small: drag again"
                else:
                    self.region, self.cells, self.drawing_region = region, [None] * len(CELLS), False
                    self.save()
                    self.select(0)
                    self.message = "new region: counts start again at cell 1"
        elif event == cv2.EVENT_LBUTTONDOWN:
            cell = self.cell_at(ix, iy)
            if cell is not None:
                self.select(cell)
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
        armed, self.redraw_armed = self.redraw_armed, False
        if self.choosing_reason:  # after s: 1-5 picks the reason, anything else cancels
            self.choosing_reason = False
            if c in REASONS:
                try:
                    set_frame_status(self.frames_csv, self.frame.name, UNREADABLE, REASONS[c])
                except OSError as exc:
                    self.message = f"could not write frames.csv ({exc.strerror}); close it in other programs"
                    return True
                done = self.frame.name
                self.step_frame(+1)
                if self.frame.name == done:
                    self.step_frame(-1)
                self.message = f"{done} marked unreadable: left out of the evaluation"
            else:
                self.message = "not marked"
            return True
        if c.isdigit():
            if len(self.entry) < 4:
                self.entry += c
        elif key == BACKSPACE:
            self.entry = self.entry[:-1]
        elif key in ENTER:
            self.commit_entry()
        elif c == "q" or key == ESC:
            return False
        elif c == "b":
            self.select(max(0, self.cell - 1))
        elif c == "z":
            self.cell_view = not self.cell_view
            self.focus()
        elif c == "c":
            self.enhanced = not self.enhanced
        elif c == "n":
            self.step_frame(+1)
        elif c == "p":
            self.step_frame(-1)
        elif c == "s":
            self.choosing_reason = True
            self.message = "unreadable because: " + "  ".join(f"{k} {v}" for k, v in REASONS.items()) + \
                "  (other key = cancel)"
        elif c == "r":
            if armed or all(v is None for v in self.cells):
                self.drawing_region, self.cell_view = True, False
                self.focus()
                self.message = "drag the new counting region with the left button"
            else:
                self.redraw_armed = True
                self.message = "press r again to redraw the region: this frame's counts will be cleared"
        return True

    # drawing --------------------------------------------------------------------------------------

    def _display_image(self) -> np.ndarray:
        if not self.enhanced:
            return self.image
        if self._enhanced_image is None:
            self._enhanced_image = enhance(self.image)
        return self._enhanced_image

    def render(self) -> np.ndarray:
        v = self.view
        affine = np.float32([[v.scale, 0, -v.ox * v.scale], [0, v.scale, -v.oy * v.scale]])
        canvas = cv2.warpAffine(self._display_image(), affine, self.size, flags=cv2.INTER_CUBIC)
        to_win = lambda x, y: tuple(int(round(c)) for c in v.to_window(x, y))
        rx0, ry0 = to_win(*self.region[:2])
        rx1, ry1 = to_win(*self.region[2:])
        box = np.s_[max(ry0, 0):max(ry1, 0), max(rx0, 0):max(rx1, 0)]
        inside = canvas[box].copy()
        canvas = (canvas * 0.4).astype(np.uint8)  # dim what is not counted
        canvas[box] = inside
        if not self.drawing_region:
            for k, rect in enumerate(cell_rects(self.region)):
                (x0, y0), (x1, y1) = to_win(*rect[:2]), to_win(*rect[2:])
                current = k == self.cell
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 0) if current else (0, 255, 255), 3 if current else 1)
                value = self.cells[k]
                label = f"{k + 1}: " + (f"{self.entry}_" if current and (self.entry or value is None) else
                                        "?" if value is None else str(value))
                cv2.putText(canvas, label, (x0 + 6, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, label, (x0 + 6, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                            cv2.LINE_AA)
        if self._drag:
            region = roi_from_drag(*self._drag, *self.image.shape[1::-1], min_size=0)
            cv2.rectangle(canvas, to_win(*region[:2]), to_win(*region[2:]), (255, 255, 0), 2)
        usable = self.usable()
        position = usable.index(self.frame) + 1 if self.frame in usable else "-"
        entered = [c for c in self.cells if c is not None]
        line = (f"{self.frame.name} [{position}/{len(usable)}] {self.shot_type.get(self.frame.name, '')}   "
                f"cell {self.cell + 1}/{len(self.cells)}   count: {self.entry}_   frame total so far: {sum(entered)} "
                f"({len(entered)}/{len(self.cells)} cells)   zoom {v.scale / self.fit_scale:.1f}x"
                f"{'   enhanced' if self.enhanced else ''}")
        bars = np.zeros((TOP, self.size[0], 3), np.uint8)
        for i, text in enumerate((line + (f"   | {self.message}" if self.message else ""), HELP)):
            cv2.putText(bars, text, (8, i * BAR + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bars, canvas])

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


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=Path, default=DATA_DIR / "eval" / "frames" / "kumbh_eval")
    p.add_argument("--counts", type=Path, default=DATA_DIR / "eval" / "annotations" / "kumbh_eval" / COUNTS_NAME)
    p.add_argument("--window", default="1600x900", help="maximum window size, WxH")
    args = p.parse_args(argv)
    print(__doc__)
    points = sorted(f.name for f in args.counts.parent.glob("*.json")) if args.counts.parent.exists() else []
    if points:
        print(f"Note: point annotations exist for {', '.join(points)}. Once a frame is counted here, counts.csv "
              "is used for it and its point file is ignored.\n")
    win_w, win_h = (int(v) for v in args.window.lower().split("x"))
    tool = CountTool(args.frames, args.counts, (win_w, win_h))
    tool.run()
    rows = read_counts(args.counts)
    usable = {f.name for f in tool.usable()}
    complete = [r for name, r in rows.items() if r.get("complete") == "yes" and name in usable]
    print(f"{len(complete)}/{len(usable)} usable frames counted, {sum(int(r['total']) for r in complete)} heads "
          f"-> {args.counts}")


if __name__ == "__main__":
    main()
