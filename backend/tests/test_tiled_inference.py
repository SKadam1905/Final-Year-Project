"""Optional tiled / upscaled inference (app.tiled_inference): tile layout, merging, sliced detection, upscaling."""

import cv2
import numpy as np
import pytest

from app.tiled_inference import merge_across_sources, sliced_detect, tile_starts, tiles, upscaled_density
from ml.kumbh_eval import density_in_region
from tests.fakes import FakeEstimator


def test_tile_layout_covers_the_frame_with_overlap():
    assert tile_starts(640, 320, 0.2) == [0, 256, 320]
    assert tile_starts(360, 320, 0.2) == [0, 40]
    assert tile_starts(300, 320, 0.2) == [0]
    rects = tiles(640, 360)
    assert len(rects) == 6 and rects[0] == (0, 0, 320, 320) and rects[-1] == (320, 40, 320, 320)
    covered = np.zeros((360, 640), bool)
    for x, y, w, h in rects:
        covered[y:y + h, x:x + w] = True
    assert covered.all()
    assert tiles(200, 100) == [(0, 0, 200, 100)]  # smaller than a tile: one clipped tile


def test_merge_only_across_sources_by_intersection_over_smaller():
    boxes = np.array([[0, 0, 20, 40],     # 0: tile A, a person
                      [1, 0, 21, 40],     # 1: tile B, the same person: merged into 0 (or 0 into 1)
                      [10, 0, 21, 40],    # 2: tile C, the half of 0 cut by a tile edge: inside 0 -> dropped
                      [5, 5, 25, 45],     # 3: tile A again, a second, overlapping person in the same tile: kept
                      [100, 0, 120, 40]])  # 4: tile B, someone else: kept
    scores = np.array([0.9, 0.8, 0.7, 0.85, 0.6])
    sources = np.array([0, 1, 2, 0, 1])
    assert merge_across_sources(boxes, scores, sources).tolist() == [0, 3, 4]


def square_detector(images):
    """Stand-in for YOLO: every white square is a person (connected components)."""
    out = []
    for img in images:
        n, _, stats, _ = cv2.connectedComponentsWithStats((img[..., 0] > 128).astype(np.uint8))
        boxes = np.array([[x, y, x + w, y + h] for x, y, w, h, area in stats[1:] if area > 20], np.float64)
        out.append((boxes.reshape(-1, 4), np.full(len(boxes), 0.9)))
    return out


def test_sliced_detect_maps_back_and_counts_each_square_once():
    frame = np.zeros((360, 640, 3), np.uint8)
    squares = [(20, 20), (250, 100), (300, 300), (500, 40), (330, 150)]  # 250..262 and 300..312 cross tile edges
    for x, y in squares:
        frame[y:y + 12, x:x + 12] = 255
    det = sliced_detect(square_detector, frame, whole=None, tile=320, overlap=0.2, upscale_to=640)
    assert det.count == len(squares)
    found = sorted(tuple(np.round(b[:2]).astype(int)) for b in det.boxes)
    assert found == sorted(squares)  # coordinates back in frame pixels
    with_whole = sliced_detect(square_detector, frame, whole=square_detector([frame])[0])
    assert with_whole.count == len(squares)  # the whole-frame boxes duplicate the tiles' and are merged


def test_upscaled_density_counts_the_same_region():
    frame = np.zeros((90, 160, 3), np.uint8)
    for factor in (1, 2, 3):
        result = upscaled_density(FakeEstimator(count=50), frame, factor)
        assert result.density.shape == (90 * factor // 8, 160 * factor // 8)
        assert result.count == pytest.approx(50)
        assert density_in_region(result.density, (160, 90), (0, 0, 80, 90)) == pytest.approx(25, rel=0.07)
