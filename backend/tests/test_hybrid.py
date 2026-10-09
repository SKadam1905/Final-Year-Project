"""P1 hybrid count (YOLO in sparse scenes, CSRNet otherwise) and P2 zone-wise grid with a hotspot."""

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.calibration import ZoneGeometry, cell_label
from app.config import ZoneConfig, get_settings
from app.detector import load_detector
from app.main import create_app
from app.pipeline import ZoneAnalyzer, choose_count
from tests.fakes import FakeDetector, FakeEstimator, make_services

FRAMES = [np.full((64, 96, 3), 90, np.uint8)] * 5  # motionless: turbulence 0


@pytest.mark.parametrize("density_count, detection_count, rule, expected", [
    (30.0, None, "agree", "density"),  # no detector
    (30.0, 12, "agree", "detection"),  # sparse: both models below 60
    (59.9, 59, "agree", "detection"),
    (30.0, 60, "agree", "density"),  # 60 is not fewer than 60
    (500.0, 5, "agree", "density"),  # YOLO misses small/hidden people in a dense crowd
    (500.0, 5, "detector", "detection"),  # the proposal's rule: YOLO's count alone decides
    (30.0, 80, "detector", "density"),
])
def test_switching_rule(density_count, detection_count, rule, expected):
    assert choose_count(density_count, detection_count, 60, rule) == expected


def analyzer(count, detector, zone=None, **settings):
    zone = zone or ZoneConfig("Z", "Zone", "x.mp4", area_m2=100.0)
    return ZoneAnalyzer(zone, FakeEstimator(count), replace(get_settings(), **settings), detector=detector)


def test_sparse_scene_uses_the_detection_count():
    status = analyzer(30, FakeDetector(12)).analyze(FRAMES, 1.0).status
    assert status.count_method == "detection" and status.count == 12
    assert status.detection_count == 12 and status.density_count == pytest.approx(30, rel=1e-4)
    assert status.density == pytest.approx(12 / 100.0)


def test_dense_scene_uses_the_density_count():
    status = analyzer(700, FakeDetector(5)).analyze(FRAMES, 1.0).status
    assert status.count_method == "density" and status.count == pytest.approx(700, rel=1e-4)
    assert status.detection_count == 5 and status.density == pytest.approx(7.0, rel=1e-4)


def test_threshold_and_rule_are_configurable():
    assert analyzer(30, FakeDetector(12), hybrid_threshold=10).analyze(FRAMES, 1.0).status.count_method == "density"
    assert analyzer(700, FakeDetector(5), hybrid_rule="detector").analyze(FRAMES, 1.0).status.count_method == "detection"
    with pytest.raises(ValueError, match="HYBRID_RULE"):
        analyzer(30, FakeDetector(1), hybrid_rule="sometimes")


def test_without_a_detector_counts_come_from_csrnet():
    status = analyzer(30, None).analyze(FRAMES, 1.0).status
    assert status.count_method == "density" and status.detection_count is None


def test_only_people_standing_inside_the_roi_are_counted():
    zone = ZoneConfig("Z", "Zone", "x.mp4", roi=((0, 0), (48, 0), (48, 64), (0, 64)), area_m2=50.0)  # left half
    boxes = [[5, 10, 15, 30], [20, 20, 30, 40], [60, 10, 70, 30]]  # feet at x = 10, 25 (in) and 65 (out)
    status = analyzer(10, FakeDetector(boxes=boxes), zone=zone).analyze(FRAMES, 1.0).status
    assert status.count_method == "detection" and status.count == 2 and status.density == pytest.approx(2 / 50)


def test_grid_counts_from_the_density_map_and_hotspot():
    density = np.zeros((9, 12), np.float32)
    density[0:3, 8:12] = 2.0  # all 24 people in the top-right ninth of a 96 x 72 frame
    density[4, 5] = 3.0  # 3 in the centre
    grid = ZoneGeometry(area_m2=90.0).grid(density, (96, 72), 3, 3)
    assert grid.counts.sum() == pytest.approx(27) and grid.counts[0, 2] == pytest.approx(24)
    assert grid.density[0, 2] == pytest.approx(24 / 10.0)  # each cell gets a ninth of 90 m²
    assert grid.hotspot() == {"row": 0, "col": 2, "label": "top-right", "count": 24.0, "density": 2.4}
    assert grid.to_dict()["counts"][1][1] == 3.0


def test_grid_counts_from_detected_feet():
    feet = np.array([[10, 70], [12, 71], [50, 30]], np.float32)  # two bottom-left, one centre
    grid = ZoneGeometry(area_m2=90.0).grid(np.zeros((9, 12), np.float32), (96, 72), 3, 3, points=feet)
    assert grid.counts[2, 0] == 2 and grid.counts[1, 1] == 1 and grid.hotspot()["label"] == "bottom-left"


def test_an_empty_view_has_no_hotspot():
    grid = ZoneGeometry(area_m2=90.0).grid(np.zeros((9, 12), np.float32), (96, 72), 3, 3)
    assert grid.hotspot() is None


def test_cell_labels():
    assert [cell_label(r, c, 3, 3) for r, c in ((0, 0), (1, 1), (2, 1))] == ["top-left", "centre", "bottom-centre"]
    assert cell_label(0, 3, 4, 4) == "row 1, column 4"


def test_payload_storage_and_alert_carry_method_grid_and_hotspot():
    services = make_services(count=700, detector=FakeDetector(5))  # 7 /m²: RED, density count
    events = services.worker.step()
    update = next(p for name, p in events if name == "density_update" and p["zone_id"] == "RAMKUND_NORTH")
    assert update["count_method"] == "density" and update["detection_count"] == 5
    assert update["grid"]["rows"] == 3 and len(update["grid"]["counts"]) == 3
    assert update["hotspot"]["label"] in {"top-left", "centre", "bottom-right", "top-centre", "top-right",
                                          "middle-left", "middle-right", "bottom-left", "bottom-centre"}
    history = services.store.history("RAMKUND_NORTH")
    assert history[-1]["count_method"] == "density" and history[-1]["hotspot"] == update["hotspot"]
    alert = next(p for name, p in events if name == "alert" and p["zone_id"] == "RAMKUND_NORTH")
    assert "Hotspot: " in alert["message"]
    assert services.worker.frames["RAMKUND_NORTH"].boxes.shape == (5, 4)


def test_detections_overlay_endpoint():
    services = make_services(count=30, detector=FakeDetector(4))
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=False)
    with TestClient(app) as client:
        client.post("/api/zones/RAMKUND_NORTH/infer")
        response = client.get("/api/zones/RAMKUND_NORTH/detections.png")
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
        health = client.get("/health").json()
        assert health["detector"]["model"] == "fake-yolo" and health["hybrid"] == {"threshold": 60.0, "rule": "agree"}

    services = make_services(count=30)  # no detector
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=False)
    with TestClient(app) as client:
        client.post("/api/zones/RAMKUND_NORTH/infer")
        response = client.get("/api/zones/RAMKUND_NORTH/detections.png")
        assert response.status_code == 503 and "detection" in response.json()["detail"]


def test_load_detector_explains_why_it_is_off(tmp_path):
    assert load_detector(None)[1].startswith("detection is off")
    detector, error = load_detector(tmp_path / "yolov8s.pt")
    assert detector is None and "not found" in error and "get_yolo_weights" in error
