import csv

import numpy as np
import pytest

from ml.annotations import load_points_json
from tests.conftest import write_video
from tests.test_annotations import CVAT_XML
from tools.annotate_points import View, nearest_point
from tools.cvat_to_points import main as cvat_main
from tools.extract_frames import extract_frames


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
