import numpy as np
import pytest

from app.calibration import (
    ZoneGeometry,
    cell_ground_areas,
    compute_homography,
    ground_area_m2,
    ground_polygon,
    is_convex,
    is_simple_polygon,
    points_in_polygon,
    polygon_area,
    project,
    rectangle_ground_points,
    rescale_homography,
    roi_mask,
)

SQUARE_PX = [(0, 0), (100, 0), (100, 100), (0, 100)]
SQUARE_M = [(0, 0), (1, 0), (1, 1), (0, 1)]
# A 4 m x 10 m walkway seen in perspective: the far end (top) looks narrower.
TRAPEZOID_PX = [(300, 400), (500, 400), (700, 800), (100, 800)]
TRAPEZOID_M = [(0, 10), (4, 10), (4, 0), (0, 0)]


def test_homography_maps_markers():
    H = compute_homography(TRAPEZOID_PX, TRAPEZOID_M)
    ground, w = project(H, np.array(TRAPEZOID_PX, float))
    np.testing.assert_allclose(ground, TRAPEZOID_M, atol=1e-6)
    assert (w > 0).all()


def test_homography_needs_four_points():
    with pytest.raises(ValueError):
        compute_homography(SQUARE_PX[:3], SQUARE_M[:3])


def test_scale_only_cell_areas():
    H = compute_homography(SQUARE_PX, SQUARE_M)  # 1 px = 1 cm
    areas = cell_ground_areas(H, (800, 800), (100, 100))  # 8x8 px cells
    np.testing.assert_allclose(areas, 0.08 * 0.08)
    assert np.nansum(areas) == pytest.approx(64.0)


def test_perspective_far_cells_are_larger_and_horizon_is_nan():
    H = compute_homography(TRAPEZOID_PX, TRAPEZOID_M)
    areas = cell_ground_areas(H, (800, 1000), (125, 100))  # 8x8 px cells
    # The sides converge at y = 200 px, so rows above it are beyond the horizon.
    assert np.isnan(areas[:24]).all()
    assert np.isfinite(areas[26:]).all()
    assert areas[60, 50] > areas[120, 50]


def test_trapezoid_roi_area_matches_ground_truth():
    H = compute_homography(TRAPEZOID_PX, TRAPEZOID_M)
    grid = (250, 200)  # 4x4 px cells
    areas = cell_ground_areas(H, (800, 1000), grid)
    mask = roi_mask(TRAPEZOID_PX, (800, 1000), grid)
    assert areas[mask].sum() == pytest.approx(40.0, rel=0.03)


def test_points_in_polygon():
    inside = points_in_polygon(np.array([[50, 50], [150, 50], [0.5, 99], [-1, 50]]), SQUARE_PX)
    assert inside.tolist() == [True, False, True, False]


def test_roi_mask_defaults_to_everything():
    assert roi_mask(None, (64, 48), (6, 8)).all()


def test_zone_geometry_with_area_only():
    density = np.full((90, 160), 0.01, np.float32)  # 144 people over a 1280x720 frame
    count, per_m2 = ZoneGeometry(area_m2=100.0).measure(density, (1280, 720))
    assert count == pytest.approx(144.0, rel=1e-5)
    assert per_m2 == pytest.approx(1.44, rel=1e-5)

    left_half = [(0, 0), (640, 0), (640, 720), (0, 720)]
    count, per_m2 = ZoneGeometry(roi=left_half, area_m2=50.0).measure(density, (1280, 720))
    assert count == pytest.approx(72.0, rel=1e-5)
    assert per_m2 == pytest.approx(1.44, rel=1e-5)


def test_zone_geometry_with_calibration():
    H = compute_homography(TRAPEZOID_PX, TRAPEZOID_M)
    density = np.zeros((125, 100), np.float32)
    density[70:100, 30:70] = 0.05  # 60 people, all inside the walkway (y 560-800 px)
    density[0:10, :] = 1.0  # beyond the horizon and outside the ROI: must be ignored
    geometry = ZoneGeometry(roi=TRAPEZOID_PX, homography=H)
    count, per_m2 = geometry.measure(density, (800, 1000))
    assert count == pytest.approx(60.0, rel=1e-4)
    assert per_m2 == pytest.approx(60.0 / 40.0, rel=0.05)


def test_zone_geometry_requires_area_or_homography():
    with pytest.raises(ValueError):
        ZoneGeometry()


def test_rectangle_ground_points_go_round_from_a():
    assert rectangle_ground_points(4, 10) == ((0, 0), (4, 0), (4, 10), (0, 10))
    with pytest.raises(ValueError):
        rectangle_ground_points(4, -1)


def test_polygon_area_and_the_exact_ground_area_of_an_image_polygon():
    assert polygon_area(SQUARE_M) == 1.0 and polygon_area(SQUARE_M[::-1]) == 1.0
    assert polygon_area([(0, 0), (4, 0), (0, 3)]) == 6.0
    H = compute_homography(TRAPEZOID_PX, TRAPEZOID_M)
    assert ground_area_m2(H, TRAPEZOID_PX) == pytest.approx(40.0, rel=1e-9)  # exact, unlike the cell sums
    near_half = [(200, 600), (600, 600), (700, 800), (100, 800)]  # image y 600-800 = ground y 0-~3.3 m
    assert 0 < ground_area_m2(H, near_half) < 20
    with pytest.raises(ValueError, match="horizon"):
        ground_polygon(H, [(300, 400), (500, 400), (400, 100)])  # y = 100 px is above the horizon (200 px)


def test_convex_and_simple_polygons():
    a, b, c, d = TRAPEZOID_PX
    assert is_convex([a, b, c, d]) and is_convex([d, c, b, a])
    assert not is_convex([a, c, b, d])  # corners clicked across the diagonal
    assert not is_convex([(0, 0), (1, 0), (2, 0), (1, 1)])  # three in a line
    star = [(0, 10), (6, -8), (-9, 3), (9, 3), (-6, -8)]
    assert not is_convex(star) and not is_simple_polygon(star)
    assert is_simple_polygon([(0, 0), (4, 0), (4, 4), (2, 1), (0, 4)])  # concave is fine
    assert not is_simple_polygon([a, c, b, d])
    assert not is_simple_polygon([(0, 0), (1, 1)])
    assert not is_simple_polygon([(0, 0), (4, 0), (0, 0), (0, 4)])  # a repeated vertex
    assert not is_simple_polygon([(0, 0), (4, 0), (4, 4), (4, 2), (0, 4)])  # an edge doubling back on itself


def test_rescaled_homography_maps_the_same_ground():
    H = compute_homography(TRAPEZOID_PX, TRAPEZOID_M)  # for an 800 x 1000 frame
    H2 = rescale_homography(H, (800, 1000), (400, 500))
    np.testing.assert_allclose(project(H2, np.array(TRAPEZOID_PX) / 2)[0], TRAPEZOID_M, atol=1e-9)


def test_zone_geometry_follows_the_frame_size():
    H = compute_homography(TRAPEZOID_PX, TRAPEZOID_M)
    geometry = ZoneGeometry(roi=TRAPEZOID_PX, homography=H, frame_size=(800, 1000))
    assert geometry.ground_area((800, 1000), (250, 200)) == pytest.approx(40.0, rel=0.03)
    assert geometry.ground_area((1600, 2000), (250, 200)) == pytest.approx(40.0, rel=0.03)  # 2x frames
    assert geometry.roi_at((800, 1000)) is geometry.roi
    assert geometry.roi_at((1600, 2000))[2] == (1400.0, 1600.0)
    feet = np.array([[800.0, 1500.0]])  # mid-walkway in a 1600 x 2000 frame; outside the 800 x 1000 one
    assert geometry.in_zone(feet, (1600, 2000)).all() and not geometry.in_zone(feet).any()
    # Without a frame size (as before calibration recorded one) nothing is rescaled.
    plain = ZoneGeometry(roi=TRAPEZOID_PX, homography=H)
    assert plain.roi_at((1600, 2000)) is plain.roi
