import numpy as np
import pytest

from ml.optical_flow import (
    burst_turbulence,
    compute_flow,
    compute_flows,
    find_cuts,
    turbulence_from_flows,
    turbulence_to_label,
    visualise_flow,
    visualise_vectors,
)
from tests.conftest import texture


def flow_field(h, w, u, v):
    flow = np.zeros((h, w, 2), np.float32)
    flow[..., 0], flow[..., 1] = u, v
    return flow


def test_uniform_motion_is_not_turbulent():
    result = turbulence_from_flows([flow_field(64, 128, 2.0, 0.0)])
    assert result.score == pytest.approx(0.0, abs=1e-9)
    assert result.valid_blocks == 8
    assert result.mean_speed == pytest.approx(2.0)
    assert result.moving_fraction == pytest.approx(1.0)


def test_interleaved_counter_flow_is_turbulent():
    flow = flow_field(64, 128, 2.0, 0.0)
    flow[::2, :, 0] = -2.0  # alternate rows move in opposite directions
    assert turbulence_from_flows([flow]).score == pytest.approx(1.0, abs=1e-9)


def test_side_by_side_orderly_streams_are_not_turbulent():
    flow = flow_field(64, 128, 2.0, 0.0)
    flow[:, 64:, 0] = -2.0  # two lanes, boundary on a block edge
    assert turbulence_from_flows([flow]).score == pytest.approx(0.0, abs=1e-9)


def test_speed_variation_in_one_direction_scores_low():
    flow = flow_field(64, 64, 1.0, 0.0)
    flow[::2, :, 0] = 3.0
    # 1 - |mean v|² / mean |v|² = 1 - 2² / ((1 + 9) / 2) = 0.2
    assert turbulence_from_flows([flow]).score == pytest.approx(0.2)


def test_static_crowd_scores_zero():
    result = turbulence_from_flows([flow_field(64, 64, 0.1, -0.1)])
    assert result.score == 0.0
    assert result.valid_blocks == 0
    assert result.moving_fraction == 0.0


def test_mask_limits_scoring_to_roi():
    flow = flow_field(64, 128, 2.0, 0.0)
    flow[::2, 64:, 0] = -2.0  # right half turbulent
    mask = np.zeros((64, 128), bool)
    mask[:, :64] = True
    assert turbulence_from_flows([flow], mask).score == pytest.approx(0.0, abs=1e-9)
    assert turbulence_from_flows([flow]).score == pytest.approx(0.5, abs=1e-9)


def test_no_flows_gives_zero():
    result = turbulence_from_flows([])
    assert (result.score, result.pairs_used) == (0.0, 0)


def shifted_burst(base, dx, dy, n=5):
    return [np.roll(base, shift=(k * dy, k * dx), axis=(0, 1)) for k in range(n)]


def test_farneback_recovers_translation():
    frames = shifted_burst(texture(), dx=2, dy=0)
    flows = compute_flows(frames)
    assert len(flows) == 4
    inner = flows[0][20:-20, 20:-20]
    assert inner[..., 0].mean() == pytest.approx(2.0, abs=0.2)
    assert abs(inner[..., 1].mean()) < 0.2


def test_scene_cut_pairs_are_skipped():
    # A looping clip: three frames of one scene, then a jump to different content.
    a, b = texture(seed=1, blur=4), texture(seed=2, blur=4)
    burst = [a, np.roll(a, 1, axis=1), np.roll(a, 2, axis=1), b, np.roll(b, 1, axis=1)]
    assert find_cuts(burst) == [False, False, True, False]
    assert len(compute_flows(burst)) == 3


def test_steady_motion_is_never_a_cut():
    burst = shifted_burst(texture(), dx=2, dy=0)
    assert find_cuts(burst) == [False] * 4
    assert find_cuts(burst[:2]) == [False]  # one pair: nothing to compare against


def test_burst_turbulence_orders_orderly_vs_chaotic_motion():
    base = texture(256, 320)
    orderly = burst_turbulence(shifted_burst(base, dx=2, dy=0))

    # Chaotic: each 16x16 tile drifts in its own direction, so every 32x32 block mixes them.
    rng = np.random.default_rng(0)
    directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
    tile_dir = rng.integers(0, 4, (256 // 16, 320 // 16))
    masks = [np.kron(tile_dir == d, np.ones((16, 16), bool)) for d in range(4)]
    chaotic_frames = []
    for k in range(5):
        frame = np.zeros_like(base)
        for d, (dx, dy) in enumerate(directions):
            frame[masks[d]] = np.roll(base, shift=(k * dy, k * dx), axis=(0, 1))[masks[d]]
        chaotic_frames.append(frame)
    chaotic = burst_turbulence(chaotic_frames)

    assert orderly.score < 0.1
    assert chaotic.score > 0.4
    assert chaotic.pairs_used == 4


def test_burst_turbulence_with_roi_polygon():
    frames = [np.dstack([f] * 3) for f in shifted_burst(texture(), dx=2, dy=0)]
    result = burst_turbulence(frames, roi=[(40, 40), (280, 40), (280, 200), (40, 200)])
    assert result.score < 0.1
    assert 0.6 < result.moving_fraction <= 1.0


def test_visualise_vectors_draws_on_a_copy():
    frame = np.zeros((64, 64, 3), np.uint8)
    out = visualise_vectors(frame, flow_field(64, 64, 2.0, 0.0), step=16)
    assert out.shape == (64, 64, 3) and out.any() and not frame.any()


def test_orderly_leftward_crowd_is_not_turbulent():
    # Angles of leftward motion sit near +180° and -180°. The std of the raw angles would be
    # ~180° (maximum "turbulence"); the vector-based score correctly stays near 0.
    rng = np.random.default_rng(0)
    flow = flow_field(64, 128, -2.0, 0.0)
    flow[..., 1] = rng.normal(0, 0.05, (64, 128))
    angles = np.arctan2(flow[..., 1], flow[..., 0])
    assert np.std(angles) / np.pi > 0.9  # what the naive angle-std formula would report
    assert turbulence_from_flows([flow]).score < 0.01


def test_compute_flow_on_still_frames():
    frames = [np.zeros((480, 640, 3), np.uint8) for _ in range(5)]
    result = compute_flow(frames)
    assert result["turbulence"] == 0.0 and result["magnitude"] == pytest.approx(0.0, abs=1e-6)
    assert len(result["flow_maps"]) == 4 and result["flow_maps"][0].shape == (480, 640, 2)


def test_compute_flow_on_moving_burst():
    frames = [np.dstack([f] * 3) for f in shifted_burst(texture(), dx=2, dy=0)]
    result = compute_flow(frames)
    assert result["magnitude"] == pytest.approx(2.0, abs=0.3)
    assert result["turbulence"] < 0.1 and result["pairs_used"] == 4


@pytest.mark.parametrize("score, label", [(0.0, "LOW"), (0.29, "LOW"), (0.3, "MEDIUM"), (0.59, "MEDIUM"),
                                          (0.6, "HIGH"), (1.0, "HIGH")])
def test_turbulence_to_label(score, label):
    assert turbulence_to_label(score) == label


def test_visualise_flow_colours_motion_and_keeps_frame_size():
    frame = np.full((120, 160, 3), 100, np.uint8)
    moving = visualise_flow(frame, flow_field(60, 80, 2.0, 0.0))
    still = visualise_flow(frame, flow_field(60, 80, 0.0, 0.0))
    assert moving.shape == frame.shape and still.shape == frame.shape
    assert np.abs(moving.astype(int) - frame).sum() > np.abs(still.astype(int) - frame).sum()
