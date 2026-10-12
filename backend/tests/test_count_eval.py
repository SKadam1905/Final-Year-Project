"""Count mode for the Kumbh evaluation frames (tools.count_eval) and its ground truth (ml.kumbh_eval)."""

import inspect

import cv2
import numpy as np
import pytest

import tools.count_eval as count_eval
from app.image_io import write_image
from ml.annotations import PointAnnotation, read_frame_status, save_points_json, set_frame_status
from ml.kumbh_eval import (boxes_in_region, cell_rects, density_in_region, load_ground_truth, read_counts, score)
from tools.count_eval import TOP, CountTool

ENTER, BACKSPACE = 13, 8


def make_eval_set(tmp_path, n=3):
    frames = tmp_path / "frames"
    rng = np.random.default_rng(0)
    for i in range(1, n + 1):
        write_image(frames / f"frame_{i:02d}.png", rng.integers(0, 255, (90, 120, 3), dtype=np.uint8))
    with open(frames / "frames.csv", "w", newline="", encoding="utf-8") as f:
        f.write("frame,shot_type,width,height\n" + "".join(
            f"frame_{i:02d}.png,{'aerial' if i == 1 else 'close'},120,90\n" for i in range(1, n + 1)))
    return frames, tmp_path / "annotations" / "counts.csv"


def window_xy(tool, x, y):
    """Window pixel (below the status lines) showing image point (x, y)."""
    wx, wy = tool.view.to_window(x, y)
    return int(wx), int(wy) + TOP


def type_count(tool, text):
    for ch in text:
        tool.handle_key(ord(ch))
    tool.handle_key(ENTER)


def test_count_cells_with_enter_zero_backspace_and_back(tmp_path):
    frames, counts = make_eval_set(tmp_path)
    tool = CountTool(frames, counts, window=(240, 180), regions={"frame_02.png": (0, 30, 120, 90)})
    assert tool.frame.name == "frame_01.png" and tool.region == (0, 0, 120, 90) and tool.cell == 0
    type_count(tool, "12")
    tool.handle_key(ENTER)  # Enter alone = 0
    for ch in "34":
        tool.handle_key(ord(ch))
    tool.handle_key(BACKSPACE)
    tool.handle_key(ENTER)  # 3
    assert tool.cells[:3] == [12, 0, 3] and tool.cell == 3
    tool.handle_key(ord("b"))
    type_count(tool, "5")  # cell 3 revised
    tool.handle_key(ord("b"))
    tool.handle_key(ord("b"))
    tool.handle_key(ENTER)  # a bare Enter on a counted cell keeps its count
    assert tool.cells[:3] == [12, 0, 5]
    tool.on_mouse(cv2.EVENT_LBUTTONDOWN, *window_xy(tool, 100, 80), 0, None)  # click: cell 9
    assert tool.cell == 8
    tool.on_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)  # a click on the status lines does nothing
    assert tool.cell == 8
    row = read_counts(counts)["frame_01.png"]
    assert (row["cell_1"], row["cell_2"], row["cell_3"], row["cell_9"]) == ("12", "0", "5", "")
    assert row["total"] == "17" and row["complete"] == "no"

    for cell in range(3, 9):
        tool.select(cell)
        type_count(tool, "1")
    row = read_counts(counts)["frame_01.png"]
    assert row["complete"] == "yes" and row["total"] == "23" and "frame complete: 23" in tool.message

    tool.handle_key(ord("n"))
    assert tool.frame.name == "frame_02.png" and tool.region == (0, 30, 120, 90)  # suggested region
    x0, y0, x1, y1 = cell_rects(tool.region)[0]
    assert (x1 - x0, y1 - y0) == (40, 20)
    canvas = tool.render()  # zoomed on cell 1; the status lines sit above the picture, within --window
    assert canvas.shape[0] == TOP + round(90 * tool.fit_scale) <= 180 and canvas.shape[1] <= 240
    assert tool.view.scale / tool.fit_scale > 1.5


def test_mark_unreadable_skips_frame_and_resume(tmp_path):
    frames, counts = make_eval_set(tmp_path)
    tool = CountTool(frames, counts, window=(240, 180))
    type_count(tool, "4")
    tool.handle_key(ord("s"))
    tool.handle_key(ord("1"))
    assert read_frame_status(frames / "frames.csv")["frame_01.png"][0] == "unreadable"
    assert tool.frame.name == "frame_02.png"
    tool.handle_key(ord("p"))
    assert tool.frame.name == "frame_02.png"  # the unreadable frame is skipped
    for _ in range(9):
        type_count(tool, "2")
    reopened = CountTool(frames, counts, window=(240, 180))
    assert reopened.frame.name == "frame_03.png"  # first usable frame not complete
    assert [f.name for f in reopened.usable()] == ["frame_02.png", "frame_03.png"]


def test_redraw_region_needs_two_presses_and_clears_counts(tmp_path):
    frames, counts = make_eval_set(tmp_path)
    tool = CountTool(frames, counts, window=(240, 180))
    type_count(tool, "7")
    tool.handle_key(ord("r"))
    assert not tool.drawing_region and "press r again" in tool.message
    tool.handle_key(ord("r"))
    assert tool.drawing_region and not tool.cell_view
    tool.on_mouse(cv2.EVENT_LBUTTONDOWN, *window_xy(tool, 10, 10), 0, None)
    tool.on_mouse(cv2.EVENT_MOUSEMOVE, *window_xy(tool, 70, 70), cv2.EVENT_FLAG_LBUTTON, None)
    tool.on_mouse(cv2.EVENT_LBUTTONUP, *window_xy(tool, 70, 70), 0, None)
    assert tool.region == pytest.approx((10.2, 10.2, 70.2, 70.2), abs=0.5)
    assert tool.cells == [None] * 9 and tool.cell == 0
    row = read_counts(counts)["frame_01.png"]
    assert float(row["x0"]) == pytest.approx(10.2, abs=0.5) and row["total"] == "0" and row["cell_1"] == ""


def test_count_mode_never_loads_a_model():
    source = inspect.getsource(count_eval)
    for name in ("csrnet", "DensityEstimator", "detector", "ultralytics", "torch"):
        assert name not in source, f"tools.count_eval must not use {name}: the ground truth stays human"


def test_ground_truth_prefers_complete_counts_and_skips_unreadable(tmp_path):
    frames, counts = make_eval_set(tmp_path, n=4)
    annotations = counts.parent
    tool = CountTool(frames, counts, window=(240, 180), regions={"frame_01.png": (0, 30, 120, 90)})
    for _ in range(9):
        type_count(tool, "3")  # frame_01 complete: 27 heads in its region
    tool.handle_key(ord("n"))
    type_count(tool, "5")  # frame_02: one cell only, incomplete
    save_points_json(annotations / "frame_01.json",  # stale points: ignored, counts.csv wins
                     PointAnnotation("frame_01.png", "eval", 120, 90, np.array([[1.0, 1.0]])))
    save_points_json(annotations / "frame_03.json",
                     PointAnnotation("frame_03.png", "eval", 120, 90, np.array([[5.0, 5.0], [100.0, 80.0]]),
                                     roi=(0, 0, 60, 45)))
    save_points_json(annotations / "frame_04.json",
                     PointAnnotation("frame_04.png", "eval", 120, 90, np.array([[5.0, 5.0]])))
    set_frame_status(frames / "frames.csv", "frame_04.png", "unreadable", "motion blur")
    truth, notes = load_ground_truth(frames, annotations)
    assert [(t.frame, t.count, t.source) for t in truth] == [("frame_01.png", 27, "counts"), ("frame_03.png", 1, "points")]
    assert truth[0].region == (0, 30, 120, 90) and truth[0].shot_type == "aerial"
    assert truth[1].region == (0, 0, 60, 45)
    text = "\n".join(notes)
    assert "frame_01.json is ignored" in text and "frame_02.png: left out, not annotated (counts incomplete)" in text
    assert "frame_04.png: left out, marked unreadable (motion blur)" in text


def test_predictions_restricted_to_the_region_and_scored():
    density = np.zeros((10, 16), np.float32)  # a 128x80 frame at 1/8
    density[5:, :] = 0.5  # 40 people in the lower half
    density[:5, :] = 0.25  # 20 in the upper half
    assert density_in_region(density, (128, 80), (0, 40, 128, 80)) == pytest.approx(40)
    assert density_in_region(density, (256, 160), (0, 80, 256, 160)) == pytest.approx(40)  # any frame scale
    boxes = np.array([[0, 0, 10, 40], [0, 50, 10, 90], [100, 30, 110, 70]])  # heads at y 2, 52, 32
    assert boxes_in_region(boxes, (0, 40, 128, 80)) == 1
    from ml.kumbh_eval import GroundTruth
    truth = [GroundTruth("a.png", "aerial", 100, (0, 0, 1, 1), "counts", 1, 1),
             GroundTruth("b.png", "close", 4, (0, 0, 1, 1), "points", 1, 1),
             GroundTruth("c.png", "close", 6, (0, 0, 1, 1), "counts", 1, 1)]
    frames, summary = score(truth, {"a.png": 80, "b.png": 6, "c.png": 6})
    assert [f["pct_error"] for f in frames] == [20.0, 50.0, 0.0]
    by_shot = {s["shot_type"]: s for s in summary}
    assert by_shot["close"]["mae"] == 1.0 and by_shot["close"]["relative_error_pct"] == 20.0
    assert by_shot["all"]["frames"] == 3 and by_shot["all"]["mae"] == pytest.approx(22 / 3, abs=0.05)
