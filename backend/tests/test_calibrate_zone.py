import json
import re

import cv2
import numpy as np
import pytest

from app.calibration import ZoneGeometry, ground_area_m2, project
from app.config import REPO_ROOT, get_settings, load_zones
from app.pipeline import ZoneAnalyzer
from tests.conftest import texture, write_video
from tests.fakes import FakeEstimator
from tools import calibration_report as report
from tools.calibrate_zone import (TOP, Calibrator, apply_calibration, calibrate, dumps_zones, ground_grid,
                                  nice_step, save_zones)

FRAME = (640, 360)
WIDTH_M, LENGTH_M = 10.0, 20.0
# A synthetic camera: the 10 m x 20 m rectangle A, B, C, D seen in perspective (far side narrower).
RECT_PX = [(220.0, 330.0), (420.0, 330.0), (380.0, 200.0), (260.0, 200.0)]
RECT_M = [(0.0, 0.0), (WIDTH_M, 0.0), (WIDTH_M, LENGTH_M), (0.0, LENGTH_M)]
G = cv2.getPerspectiveTransform(np.float32(RECT_M), np.float32(RECT_PX)).astype(np.float64)  # ground -> image
ZONE_M = [(-5.0, -2.0), (15.0, -2.0), (15.0, 30.0), (-5.0, 30.0)]  # 20 m x 32 m = 640 m²
ZONE_AREA = 640.0


def to_image(points_m):
    homog = np.column_stack([np.asarray(points_m, float), np.ones(len(points_m))]) @ G.T
    return [tuple(p) for p in homog[:, :2] / homog[:, 2:]]


ZONE_PX = to_image(ZONE_M)


def zones_file(tmp_path, zones=None):
    data = {"_comment": ["test zones"], "zones": zones or [
        {"id": "Z1", "name": "zone one", "source": "rtsp://localhost:8554/z1", "roi": None, "area_m2": 1000.0,
         "calibration": None},
        {"id": "Z2", "name": "zone two", "source": "rtsp://localhost:8554/z2", "roi": None, "area_m2": 500.0,
         "calibration": None},
    ]}
    path = tmp_path / "zones.json"
    path.write_text(dumps_zones(data), encoding="utf-8")
    return path


SOURCE = {"method": "tape", "width_m": WIDTH_M, "length_m": LENGTH_M, "measured_by": "tester", "rectangle": None}


def test_synthetic_camera_is_inside_the_frame():
    for x, y in ZONE_PX:
        assert 0 <= x < FRAME[0] and 0 <= y < FRAME[1]


# calibrate() ------------------------------------------------------------------------------------

def test_calibrate_recovers_the_true_area():
    result = calibrate(RECT_PX, WIDTH_M, LENGTH_M, ZONE_PX)
    assert result.area_m2 == pytest.approx(ZONE_AREA, rel=0.01)
    np.testing.assert_allclose(project(result.homography, np.array(RECT_PX))[0], RECT_M, atol=1e-6)
    assert result.rectangle_m2 == 200.0
    assert 1.0 < result.reach < 3.0
    assert 0 < result.click_sd_pct < 10
    assert all(round(v, 1) == v for p in result.roi + result.image_points for v in p)  # saved at 0.1 px


def test_calibrate_rejects_corners_out_of_order_and_bad_polygons():
    a, b, c, d = RECT_PX
    with pytest.raises(ValueError, match="in order"):
        calibrate([a, c, b, d], WIDTH_M, LENGTH_M, ZONE_PX)
    with pytest.raises(ValueError, match="cross itself"):
        p = ZONE_PX
        calibrate(RECT_PX, WIDTH_M, LENGTH_M, [p[0], p[2], p[1], p[3]])
    with pytest.raises(ValueError, match="horizon"):
        calibrate(RECT_PX, WIDTH_M, LENGTH_M, [(200, 300), (440, 300), (320, 2)])  # the apex is above it
    with pytest.raises(ValueError, match="positive"):
        calibrate(RECT_PX, 0, LENGTH_M, ZONE_PX)


def test_ground_grid_lies_on_the_rectangle():
    H = calibrate(RECT_PX, WIDTH_M, LENGTH_M, ZONE_PX).homography
    assert nice_step(32) == 5 and nice_step(4) == 0.5 and nice_step(900) == 100
    lines = ground_grid(H, (0, 0, WIDTH_M, LENGTH_M), 10)
    ends = {tuple(np.round(line[0]).astype(int)) for line in lines} | {
        tuple(np.round(line[-1]).astype(int)) for line in lines}
    for corner in RECT_PX:  # the 10 m lines run through the rectangle's corners
        assert tuple(int(v) for v in corner) in ends


# zones.json -------------------------------------------------------------------------------------

def test_repo_zones_json_round_trips_unchanged():
    text = (REPO_ROOT / "config" / "zones.json").read_text(encoding="utf-8")
    assert dumps_zones(json.loads(text)) == text


def test_apply_keeps_the_estimate_once_and_other_zones_untouched(tmp_path):
    path = zones_file(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    result = calibrate(RECT_PX, WIDTH_M, LENGTH_M, ZONE_PX)
    first = apply_calibration(data, "Z1", result, FRAME, SOURCE)
    z1 = first["zones"][0]
    assert z1["area_m2_estimated"] == 1000.0 and z1["roi_estimated"] is None
    assert z1["area_m2"] == round(result.area_m2, 1) and z1["frame_size"] == [640, 360]
    assert z1["roi"] == [list(p) for p in result.roi]
    assert z1["calibration"]["source"] == {"method": "tape", "width_m": 10.0, "length_m": 20.0,
                                           "measured_by": "tester"}  # empty fields dropped
    assert list(z1)[:7] == ["id", "name", "source", "roi", "area_m2", "area_m2_estimated", "roi_estimated"]
    assert first["zones"][1] == data["zones"][1] and first["_comment"] == data["_comment"]
    assert data["zones"][0]["area_m2"] == 1000.0  # the input is not modified

    smaller = calibrate(RECT_PX, WIDTH_M, LENGTH_M, ZONE_PX[:3])
    again = apply_calibration(first, "Z1", smaller, FRAME, SOURCE)["zones"][0]
    assert again["area_m2_estimated"] == 1000.0  # still the original estimate, not the first calibration
    assert again["area_m2"] == round(smaller.area_m2, 1)
    with pytest.raises(ValueError, match="unknown zone"):
        apply_calibration(data, "NOPE", result, FRAME, SOURCE)


def test_saved_calibration_loads_and_the_live_area_is_calibrated(tmp_path):
    path = zones_file(tmp_path)
    result = calibrate(RECT_PX, WIDTH_M, LENGTH_M, ZONE_PX)
    save_zones(path, apply_calibration(json.loads(path.read_text(encoding="utf-8")), "Z1", result, FRAME, SOURCE))
    text = path.read_text(encoding="utf-8")
    assert '"image_points": [[220.0, 330.0], [420.0, 330.0], [380.0, 200.0], [260.0, 200.0]]' in text

    z1, z2 = load_zones(path)
    assert (z1.area_source, z2.area_source) == ("calibrated", "estimated")
    assert z1.area_m2 == pytest.approx(ZONE_AREA, rel=0.01) and z1.area_m2_estimated == 1000.0
    assert z1.frame_size == FRAME and z1.calibration.source["method"] == "tape"
    assert z1.calibration.homography is not None

    # The live geometry divides by the calibrated area (on its density grid), not by the estimate.
    calibrated = ZoneGeometry.from_zone(z1).ground_area(FRAME, (45, 80))
    assert calibrated == pytest.approx(ZONE_AREA, rel=0.03)
    assert ZoneGeometry.from_zone(z2).ground_area(FRAME, (45, 80)) == 500.0


def test_save_refuses_a_config_that_does_not_load(tmp_path):
    path = zones_file(tmp_path)
    before = path.read_text(encoding="utf-8")
    result = calibrate(RECT_PX, WIDTH_M, LENGTH_M, ZONE_PX)
    data = apply_calibration(json.loads(before), "Z1", result, FRAME, SOURCE)
    data["zones"][0]["area_m2"] = 9999.0  # not the polygon's calibrated area
    with pytest.raises(ValueError, match="calibrated area"):
        save_zones(path, data)
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["zones.json"]  # no temporary file left


def test_load_zones_checks_the_stored_homography(tmp_path):
    result = calibrate(RECT_PX, WIDTH_M, LENGTH_M, ZONE_PX)
    data = apply_calibration(json.loads(zones_file(tmp_path).read_text(encoding="utf-8")), "Z1", result, FRAME,
                             SOURCE)
    data["zones"][0]["calibration"]["homography"][0][0] *= 1.5
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_zones(path)


def test_hand_made_calibration_without_homography_still_loads(tmp_path):
    path = zones_file(tmp_path, [{"id": "Z", "source": "rtsp://x", "roi": [list(p) for p in ZONE_PX],
                                  "calibration": {"image_points": RECT_PX, "ground_points": RECT_M}}])
    [zone] = load_zones(path)
    assert zone.calibration.homography is None and zone.frame_size is None
    assert zone.area_m2 == pytest.approx(ZONE_AREA, rel=0.01)  # filled in from the roi


def test_live_analyzer_uses_the_calibrated_area(tmp_path):
    path = zones_file(tmp_path)
    result = calibrate(RECT_PX, WIDTH_M, LENGTH_M, ZONE_PX)
    save_zones(path, apply_calibration(json.loads(path.read_text(encoding="utf-8")), "Z1", result, FRAME, SOURCE))
    z1, z2 = load_zones(path)
    frames = [np.dstack([texture(FRAME[1], FRAME[0])] * 3)] * 5
    for zone in (z1, z2):
        analyzer = ZoneAnalyzer(zone, FakeEstimator(1000.0), get_settings())
        status = analyzer.analyze(frames, 0.0).status
        area = analyzer.geometry.ground_area(FRAME, (45, 80))
        assert status.density == pytest.approx(status.count / area, rel=1e-5)
    assert ZoneAnalyzer(z1, FakeEstimator(), get_settings()).geometry.ground_area(FRAME, (45, 80)) == \
        pytest.approx(ZONE_AREA, rel=0.03)


# the window ---------------------------------------------------------------------------------------

def click(tool, x, y, event=cv2.EVENT_LBUTTONDOWN):
    wx, wy = tool.view.to_window(x, y)
    tool.on_mouse(event, int(round(wx - 0.5)), int(round(wy - 0.5)) + TOP, 0)


def type_text(tool, text):
    for ch in text:
        tool.handle_key(ord(ch))
    tool.handle_key(13)


def test_clicking_through_a_calibration_saves_the_true_area():
    saved = []
    tool = Calibrator(np.zeros((FRAME[1], FRAME[0], 3), np.uint8), "Z1", lambda r: saved.append(r) or "saved",
                      estimate=1000.0)
    for x, y in RECT_PX:
        click(tool, x, y)
    assert tool.phase == "width"
    type_text(tool, "10")
    type_text(tool, "20,0")  # a decimal comma works too
    assert (tool.phase, tool.width_m, tool.length_m) == ("polygon", 10.0, 20.0)
    click(tool, 320, 3)  # above the horizon (y = 5 px): refused
    assert tool.polygon == [] and "horizon" in tool.message
    for x, y in ZONE_PX:
        click(tool, x, y)
    assert "m2" in tool.message
    tool.handle_key(ord("w"))
    assert not saved and "finish the polygon" in tool.message
    tool.handle_key(13)
    assert tool.phase == "review" and "estimate was 1,000 m2" in tool.message
    assert tool.render().shape == (tool.size[1] + TOP, tool.size[0], 3)
    assert tool.snapshot().shape == (tool.size[1], tool.size[0], 3)
    tool.handle_key(ord("w"))
    assert tool.saved and saved[0].area_m2 == pytest.approx(ZONE_AREA, rel=0.02)


def test_out_of_order_corners_undo_and_preset_size():
    tool = Calibrator(np.zeros((FRAME[1], FRAME[0], 3), np.uint8), "Z1", lambda r: "saved",
                      width_m=WIDTH_M, length_m=LENGTH_M)
    a, b, c, d = RECT_PX
    for x, y in (a, c, b, d):
        click(tool, x, y)
    assert tool.phase == "corners" and "in order" in tool.message
    click(tool, *a)  # a fifth click is ignored
    assert len(tool.corners) == 4
    for _ in range(3):
        click(tool, 0, 0, cv2.EVENT_RBUTTONDOWN)  # right click = undo
    for x, y in (b, c, d):
        click(tool, x, y)
    assert tool.phase == "polygon"  # --width/--length given: no typing
    tool.handle_key(ord("u"))  # back to the corners
    assert tool.phase == "corners" and len(tool.corners) == 3
    tool.handle_key(ord("r"))
    assert tool.corners == [] and tool.phase == "corners"


# the report ---------------------------------------------------------------------------------------

def calibrated_config(tmp_path):
    path = zones_file(tmp_path)
    result = calibrate(RECT_PX, WIDTH_M, LENGTH_M, ZONE_PX)
    save_zones(path, apply_calibration(json.loads(path.read_text(encoding="utf-8")), "Z1", result, FRAME, SOURCE))
    return path


def test_report_area_rows(tmp_path):
    s = get_settings()
    rows = {r["zone"]: r for r in report.area_rows(load_zones(calibrated_config(tmp_path)), s)}
    z1, z2 = rows["Z1"], rows["Z2"]
    assert z2["calibrated m² (polygon)"] == "not calibrated" and z2["estimate m²"] == 500.0
    assert z1["estimate m²"] == 1000.0 and z1["estimate ROI"] == "whole frame"
    assert z1["calibrated m² (polygon)"] == pytest.approx(ZONE_AREA, rel=0.01)
    live = z1["live m² (density grid)"]
    assert live == pytest.approx(ZONE_AREA, rel=0.03)
    assert z1["estimate ÷ calibrated"] == round(1000.0 / live, 2)
    assert z1["people at YELLOW (estimate -> calibrated)"] == f"{s.risk_yellow * 1000:,.0f} -> {s.risk_yellow * live:,.0f}"
    assert z1["measured"] == "tape, tester"


def test_report_replays_a_clip_both_ways_with_one_model_run(tmp_path):
    zone = load_zones(calibrated_config(tmp_path))[0]
    clip = write_video(tmp_path / "clip.avi", [np.dstack([texture(FRAME[1], FRAME[0])] * 3)] * 50, fps=10)
    estimator = FakeEstimator(1000.0)
    calls = []
    estimator.predict = (lambda predict: lambda frame: calls.append(1) or predict(frame))(estimator.predict)
    rows = report.replay(zone, clip, get_settings(), estimator, None)
    assert len(rows) == 3 and len(calls) == 3  # bursts at 0, 2 and 4 s; CSRNet once per burst
    for row in rows:
        assert row["estimated_count"] == pytest.approx(1000.0, rel=1e-4)  # old ROI: the whole frame
        assert row["estimated_density"] == pytest.approx(1.0, rel=1e-3)  # 1000 people / 1000 m²
        assert row["calibrated_count"] < 1000.0
    summary = report.summarize("Z1", rows)
    assert summary["readings"] == 3 and re.fullmatch(r"\d+% \(\d+ up, \d+ down\)", summary["level changed"])
    assert set(summary) >= {"people counted (mean)", "persons/m² (mean)", "readings at RED+", "level changed"}


def test_report_writes_nothing_without_a_calibrated_zone(tmp_path, capsys):
    out = tmp_path / "out" / "zone_calibration"
    report.main(["--config", str(zones_file(tmp_path)), "--out", str(out)])
    assert "No zone is calibrated yet" in capsys.readouterr().out
    assert not out.parent.exists()
    report.main(["--config", str(calibrated_config(tmp_path)), "--out", str(out)])
    text = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "| Z1 | calibrated |" in text and "| Z2 | estimated |" in text
