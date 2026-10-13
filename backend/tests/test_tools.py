import csv

import cv2
import numpy as np
import pytest

from app.image_io import write_image
from ml.annotations import evaluation_frames, load_points_json, read_frame_status
from tests.conftest import write_video
from tests.test_annotations import CVAT_XML
from tools.annotate_points import (MAX_ZOOM, ZOOM_STEP, Annotator, View, enhance, nearest_point, roi_from_drag,
                                   wheel_delta)
from tools.cvat_to_points import main as cvat_main
from tools.extract_frames import extract_frames

ESC_KEY = 27


def test_view_mapping_roundtrip_and_zoom_keeps_cursor_point():
    view = View(scale=0.5, ox=100, oy=50)
    assert view.to_window(*view.to_image(30, 40)) == pytest.approx((30, 40))
    before = view.to_image(200, 150)
    view.zoom_at(2.0, 200, 150, min_scale=0.5, max_scale=4)
    assert view.scale == 1.0
    assert view.to_image(200, 150) == pytest.approx(before)
    view.clamp(image_w=1280, image_h=720, win_w=800, win_h=450)
    assert 0 <= view.ox <= 480 and 0 <= view.oy <= 270


def test_nearest_point():
    points = [(10.0, 10.0), (50.0, 50.0)]
    assert nearest_point(points, 48, 47, max_dist=5) == 1
    assert nearest_point(points, 30, 30, max_dist=5) is None
    assert nearest_point([], 0, 0, max_dist=5) is None


def test_roi_from_drag_normalises_and_clips():
    assert roi_from_drag(50, 40, 10, 5, w=64, h=48) == (10, 5, 50, 40)
    assert roi_from_drag(-5, -5, 70, 60, w=64, h=48) == (0, 0, 64, 48)
    assert roi_from_drag(10, 10, 12, 30, w=64, h=48) is None  # narrower than min_size


def test_annotator_roi_flow(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(cv2, "imshow", lambda _name, canvas: shown.append(canvas))
    frame = tmp_path / "frames" / "f.png"
    write_image(frame, np.full((80, 100, 3), 200, np.uint8))
    out = tmp_path / "annotations"
    ann = Annotator([frame], out, "vid", window=(200, 160))  # fit scale 2: image = window / 2
    click = lambda event, x, y, flags=0: ann.on_mouse(event, x, y, flags, None)

    click(cv2.EVENT_LBUTTONDOWN, 150, 150)  # head at image (75.25, 75.25), before any ROI
    ann.toggle_roi_drawing()
    click(cv2.EVENT_LBUTTONDOWN, 20, 20)
    click(cv2.EVENT_MOUSEMOVE, 100, 80, cv2.EVENT_FLAG_LBUTTON)
    click(cv2.EVENT_LBUTTONUP, 100, 80)
    assert ann.roi is None and "1 heads would be outside" in ann.message  # never drops a head silently

    ann.on_mouse(cv2.EVENT_RBUTTONDOWN, 150, 150, 0, None)
    click(cv2.EVENT_LBUTTONDOWN, 20, 20)
    click(cv2.EVENT_LBUTTONUP, 100, 80)
    assert ann.roi == pytest.approx((10.25, 10.25, 50.25, 40.25)) and not ann.drawing_roi

    click(cv2.EVENT_LBUTTONDOWN, 180, 150)  # outside the ROI: ignored
    click(cv2.EVENT_LBUTTONDOWN, 60, 60)  # inside: image (30.25, 30.25)
    saved = load_points_json(out / "f.json")
    assert saved.roi == pytest.approx((10.25, 10.25, 50.25, 40.25))
    np.testing.assert_allclose(saved.points, [[30.25, 30.25]])
    ann.render()
    assert shown[-1][50, 50, 0] == 200 and shown[-1][150, 180, 0] == 80  # [row, col]: outside the ROI is dimmed

    ann.set_roi(None)
    assert load_points_json(out / "f.json").roi is None


def test_enhance_changes_only_the_display():
    rng = np.random.default_rng(0)
    hazy = (120 + rng.integers(0, 20, (90, 160, 3))).astype(np.uint8)  # low contrast, like the aerial haze
    before = hazy.copy()
    out = enhance(hazy)
    assert out.shape == hazy.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(hazy, before)
    lap = lambda img: cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert lap(out) > 2 * lap(hazy)


def make_frames(tmp_path, n=2):
    folder = tmp_path / "frames" / "eval"
    rng = np.random.default_rng(1)
    for i in range(1, n + 1):
        write_image(folder / f"frame_{i:02d}.png", rng.integers(0, 255, (80, 100, 3), dtype=np.uint8))
    with open(folder / "frames.csv", "w", newline="", encoding="utf-8") as f:
        f.write("frame,shot_type,time_s\n" + "".join(f"frame_{i:02d}.png,aerial,{i}.5\n" for i in range(1, n + 1)))
    return folder, sorted(folder.glob("*.png"))


def test_annotator_enhance_key_and_smooth_zoom_to_6x(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(cv2, "imshow", lambda _name, canvas: shown.append(canvas))
    folder, frames = make_frames(tmp_path)
    ann = Annotator(frames, tmp_path / "ann", "eval", window=(200, 160))
    ann.on_mouse(cv2.EVENT_LBUTTONDOWN, 60, 60, 0, None)
    saved = load_points_json(tmp_path / "ann" / "frame_01.json")
    ann.render()
    ann.handle_key(ord("c"))
    ann.render()
    assert ann.enhanced and not np.array_equal(shown[0], shown[1])
    np.testing.assert_array_equal(load_points_json(tmp_path / "ann" / "frame_01.json").points, saved.points)
    ann.handle_key(ord("c"))
    assert not ann.enhanced

    for _ in range(3):
        ann.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 80, 120 << 16, None)  # wheel up
    assert ann.view.scale / ann.fit_scale == pytest.approx(ZOOM_STEP ** 3)
    for _ in range(60):
        ann.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 80, 120 << 16, None)
    assert ann.view.scale / ann.fit_scale == pytest.approx(MAX_ZOOM) == 6.0
    ann.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 80, -120 << 16, None)  # wheel down: zooms out one step
    assert ann.view.scale / ann.fit_scale == pytest.approx(MAX_ZOOM / ZOOM_STEP)


def test_wheel_delta_from_flags():
    assert wheel_delta(120 << 16) == 120 and wheel_delta(-120 << 16) == -120
    assert wheel_delta(((-120 << 16) & 0xFFFFFFFF) | 8) == -120  # unsigned flags with a modifier bit set


def test_mark_frame_unreadable_with_reason(tmp_path):
    folder, frames = make_frames(tmp_path, n=3)
    out = tmp_path / "ann"
    ann = Annotator(frames, out, "eval", window=(200, 160))
    ann.on_mouse(cv2.EVENT_LBUTTONDOWN, 60, 60, 0, None)  # a guessed head on frame 1...
    assert ann.handle_key(ord("s")) and ann.choosing_reason
    assert ann.handle_key(ord("2"))  # ...then marked unreadable: too dense
    status = read_frame_status(folder / "frames.csv")
    assert status == {"frame_01.png": ("unreadable", "heads too small or dense to separate")}
    with open(folder / "frames.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["shot_type"] for r in rows] == ["aerial"] * 3 and rows[1]["status"] == ""  # other columns kept
    assert [a.image for _, a in evaluation_frames(folder, out)] == []  # excluded, not guessed

    ann.handle_key(ord("n"))
    ann.handle_key(ord("s"))
    assert ann.handle_key(ESC_KEY) and not ann.choosing_reason  # Esc cancels the menu, does not quit
    assert "frame_02.png" not in read_frame_status(folder / "frames.csv")
    ann.on_mouse(cv2.EVENT_LBUTTONDOWN, 60, 60, 0, None)
    assert [a.image for _, a in evaluation_frames(folder, out)] == ["frame_02.png"]

    reopened = Annotator(frames, out, "eval", window=(200, 160))
    assert reopened.index == 2  # resumes after annotated and unreadable frames
    reopened.goto(0)
    assert reopened.status == "unreadable"
    reopened.handle_key(ord("s"))  # s on an unreadable frame clears the mark
    assert read_frame_status(folder / "frames.csv") == {}
    assert reopened.handle_key(ord("q")) is False


def test_extract_frames_at_requested_rate(tmp_path):
    frames = [np.full((48, 64, 3), k * 5, np.uint8) for k in range(50)]  # 2 s at 25 fps
    video = write_video(tmp_path / "clip.avi", frames, fps=25)
    out = tmp_path / "frames" / "clip"
    assert extract_frames(video, out, fps=5) == 10
    with open(out / "frames.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["time_s"] for r in rows[:3]] == ["0.00", "0.20", "0.40"]
    assert (out / "frame_0010.jpg").exists()


def test_cvat_to_points_cli(tmp_path, capsys):
    xml = tmp_path / "annotations.xml"
    xml.write_text(CVAT_XML, encoding="utf-8")
    out = tmp_path / "annotations"
    cvat_main([str(xml), "--label", "head", "--video", "pandharpur_01", "--out", str(out),
               "--frames", str(tmp_path / "frames")])
    ann = load_points_json(out / "pandharpur_01" / "frame_0001.json")
    assert len(ann.points) == 3 and ann.video == "pandharpur_01"
    assert "frames not found" in capsys.readouterr().out
