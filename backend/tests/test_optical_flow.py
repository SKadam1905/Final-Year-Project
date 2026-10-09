import cv2
import numpy as np
import pytest

from ml.optical_flow import (
    FlowParams,
    burst_has_cut,
    burst_turbulence,
    scaled_to_width,
    compute_flow,
    compute_flows,
    find_cuts,
    turbulence_from_flows,
    turbulence_to_label,
    visualise_flow,
    visualise_vectors,
)
from tests.conftest import texture, write_video


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


def test_density_weighting_ignores_motion_where_there_are_no_people():
    flow = flow_field(64, 128, 2.0, 0.0)
    flow[::2, :64, 0] = -2.0  # left half: rippling water (opposed motion), right half: an orderly crowd
    density = np.zeros((64, 128))
    density[:, 64:] = 0.3  # the density map finds people only on the right
    assert turbulence_from_flows([flow]).score == pytest.approx(0.5, abs=1e-9)
    assert turbulence_from_flows([flow], weights=density).score == pytest.approx(0.0, abs=1e-9)
    # With people everywhere (any uniform density) it is exactly the whole-frame score.
    assert turbulence_from_flows([flow], weights=np.full((64, 128), 0.07)).score == turbulence_from_flows([flow]).score


def test_density_weighting_counts_blocks_by_their_people():
    flow = flow_field(64, 128, 2.0, 0.0)
    flow[::2, :64, 0] = -2.0  # left half turbulent (T = 1), right half orderly (T = 0)
    density = np.ones((64, 128))
    density[:, :64] = 3.0  # three times as many people on the turbulent side
    assert turbulence_from_flows([flow], weights=density).score == pytest.approx(0.75, abs=1e-9)


def test_burst_turbulence_takes_a_density_map_at_any_resolution():
    frames = crowd_scene(chaotic=True)
    uniform = np.full((30, 40), 0.2, np.float32)  # e.g. CSRNet's 1/8-resolution map
    assert burst_turbulence(frames, density=uniform).score == pytest.approx(burst_turbulence(frames).score, abs=1e-6)
    nobody = np.zeros((30, 40), np.float32)
    assert burst_turbulence(frames, density=nobody).score == 0.0


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


def chaotic_burst(base, tile=16, step=2):
    """Each tile x tile patch drifts in its own direction, so every turbulence block mixes several."""
    rng = np.random.default_rng(0)
    directions = [(step, 0), (-step, 0), (0, step), (0, -step)]
    h, w = base.shape
    tile_dir = rng.integers(0, 4, (h // tile, w // tile))
    masks = [np.kron(tile_dir == d, np.ones((tile, tile), bool)) for d in range(4)]
    frames = []
    for k in range(5):
        frame = np.zeros_like(base)
        for d, (dx, dy) in enumerate(directions):
            frame[masks[d]] = np.roll(base, shift=(k * dy, k * dx), axis=(0, 1))[masks[d]]
        frames.append(frame)
    return frames


def test_burst_turbulence_orders_orderly_vs_chaotic_motion():
    # 640 px wide: the turbulence block is 32 px (16 px at the default 320 px flow width),
    # so each block holds four 16 px tiles moving different ways.
    base = texture(512, 640)
    orderly = burst_turbulence(shifted_burst(base, dx=2, dy=0))
    chaotic = burst_turbulence(chaotic_burst(base))

    assert orderly.score < 0.1
    assert chaotic.score > 0.4
    assert chaotic.pairs_used == 4


def test_flow_is_reported_in_full_resolution_pixels():
    burst = shifted_burst(texture(360, 1280, blur=3), dx=4, dy=0)  # 4 px/frame at 1280 px wide
    for width in (1280, 640, 320):
        result = compute_flow(burst, flow_params=FlowParams(max_width=width))
        assert result["flow_maps"][0].shape[1] == width
        assert result["magnitude"] == pytest.approx(4.0, rel=0.15), width


def test_turbulence_agrees_across_flow_widths():
    base = texture(512, 640)
    chaotic = chaotic_burst(base)
    orderly = shifted_burst(base, dx=2, dy=0)
    for burst in (chaotic, orderly):
        full = burst_turbulence(burst, flow_params=FlowParams(max_width=640)).score
        reduced = burst_turbulence(burst, flow_params=FlowParams(max_width=320)).score
        assert reduced == pytest.approx(full, abs=0.1)


def test_window_and_block_scale_with_flow_width():
    assert [scaled_to_width(32, w, minimum=4) for w in (1280, 640, 320, 240, 40)] == [64, 32, 16, 12, 4]
    assert scaled_to_width(15, 320, minimum=5) == 8


def test_burst_turbulence_with_roi_polygon():
    # The whole frame shifting is what a panning camera looks like; with compensation on, nothing
    # would move relative to the ground. Switched off here to test the ROI alone.
    frames = [np.dstack([f] * 3) for f in shifted_burst(texture(), dx=2, dy=0)]
    result = burst_turbulence(frames, roi=[(40, 40), (280, 40), (280, 200), (40, 200)],
                              flow_params=FlowParams(compensate_camera=False))
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
    assert len(result["flow_maps"]) == 4 and result["flow_maps"][0].shape == (240, 320, 2)  # FLOW_MAX_WIDTH


def test_compute_flow_on_moving_burst():
    frames = [np.dstack([f] * 3) for f in shifted_burst(texture(), dx=2, dy=0)]
    result = compute_flow(frames)
    assert result["magnitude"] == pytest.approx(2.0, abs=0.3)
    assert result["turbulence"] < 0.1 and result["pairs_used"] == 4


def crowd_scene(pan=(0, 0), chaotic=False, n=5, size=(320, 240), people=150, speed=2.0, seed=0):
    """A textured ground with people (discs) on it, half walking (all one way, or each its own way
    when `chaotic`) and half standing, filmed by a camera panning `pan` px/frame."""
    rng = np.random.default_rng(seed)
    w, h = size
    ground = cv2.cvtColor(texture(h + 200, w + 200, seed=seed + 1, blur=3), cv2.COLOR_GRAY2BGR)
    pos = rng.uniform((100, 100), (100 + w, 100 + h), (people, 2))
    angle = rng.uniform(0, 2 * np.pi, people) if chaotic else np.zeros(people)
    vel = speed * np.column_stack([np.cos(angle), np.sin(angle)])
    vel[people // 2:] = 0
    frames = []
    for k in range(n):
        cx, cy = 100 + pan[0] * k, 100 + pan[1] * k
        frame = ground[cy:cy + h, cx:cx + w].copy()
        for x, y in pos + vel * k:
            centre = (int(round(x - cx)), int(round(y - cy)))
            cv2.circle(frame, centre, 7, (40, 40, 60), -1)
            cv2.circle(frame, centre, 3, (200, 210, 230), -1)
        frames.append(frame)
    return frames


RAW = FlowParams(compensate_camera=False)


@pytest.mark.parametrize("chaotic", [False, True])
def test_panning_camera_is_removed_so_turbulence_matches_a_fixed_camera(chaotic):
    fixed = burst_turbulence(crowd_scene(chaotic=chaotic)).score
    for pan in [(3, 0), (2, 2), (-3, 1)]:
        frames = crowd_scene(pan, chaotic)
        result = burst_turbulence(frames)
        assert result.moving_camera and result.camera_motion == pytest.approx(np.hypot(*pan), abs=0.4)
        assert result.score == pytest.approx(fixed, abs=0.03)
        if chaotic:  # uncorrected, the pan's common motion makes pushing look orderly
            assert burst_turbulence(frames, flow_params=RAW).score < fixed / 2
    if not chaotic:
        assert fixed < 0.2  # an orderly crowd stays low however the camera moves


def test_slow_whole_frame_motion_is_not_taken_for_a_camera():
    # A crowd moving as one block over the whole view looks like a slow pan: below 1 px/frame
    # (at 640 px wide) it is left alone, so a fixed camera over a procession scores as before.
    base = cv2.cvtColor(texture(360, 640, blur=3), cv2.COLOR_GRAY2BGR)

    def drift(px_per_frame):  # the whole view moving right, sub-pixel steps by interpolation
        return [cv2.warpAffine(base, np.float32([[1, 0, k * px_per_frame], [0, 1, 0]]), (640, 360),
                               borderMode=cv2.BORDER_REFLECT) for k in range(5)]

    slow = burst_turbulence(drift(0.6))
    assert not slow.moving_camera and slow.camera_motion == pytest.approx(0.6, abs=0.15)  # measured, not removed
    assert slow.score == burst_turbulence(drift(0.6), flow_params=RAW).score
    fast = burst_turbulence(drift(1.5))
    assert fast.moving_camera and fast.camera_motion == pytest.approx(1.5, abs=0.2)


def test_a_zone_marked_fixed_camera_never_removes_motion():
    from app.config import ZoneConfig
    from app.pipeline import ZoneAnalyzer
    from tests.fakes import FakeEstimator

    analyzer = ZoneAnalyzer(ZoneConfig("Z", "Z", "x", area_m2=100.0, fixed_camera=True), FakeEstimator())
    assert not analyzer.flow_params.compensate_camera
    result = analyzer.analyze(crowd_scene(pan=(3, 0), chaotic=True), 0.0)
    assert not result.turbulence.moving_camera and result.turbulence.camera_motion == 0.0


def test_fixed_camera_turbulence_is_unchanged(tmp_path):
    """A static camera (here through a compressed video file) scores exactly as without compensation."""
    for chaotic in (False, True):
        cap = cv2.VideoCapture(str(write_video(tmp_path / f"static_{chaotic}.avi", crowd_scene(chaotic=chaotic, n=5))))
        frames = [cap.read()[1] for _ in range(5)]
        with_compensation, without = burst_turbulence(frames), burst_turbulence(frames, flow_params=RAW)
        assert with_compensation.score == without.score and with_compensation.mean_speed == without.mean_speed
        assert with_compensation.camera_motion < 0.1 and not with_compensation.moving_camera


def test_a_colour_change_is_a_cut_even_when_grey_levels_match():
    base = texture(blur=4)
    tint = lambda b, g, r: np.dstack([base * b, base * g, base * r]).astype(np.uint8)  # noqa: E731
    reddish, greenish = tint(0.30, 0.45, 1.6).clip(0, 255), tint(0.45, 1.1, 0.30)
    burst = [reddish, reddish, greenish, greenish, greenish]
    assert find_cuts(burst) == [False, True, False, False]
    assert burst_has_cut(burst) and not burst_has_cut([reddish] * 5)
    assert burst_turbulence(burst).scene_cut


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
