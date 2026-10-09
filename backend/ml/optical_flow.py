"""Burst optical flow and a crowd-turbulence score (plan §4.2).

Farneback dense flow assumes small motion between frames, so it runs on bursts of 5
consecutive frames (40 ms apart at 25 fps; the burst spans ~0.2 s), never on frames
seconds apart, where the vectors are garbage.

Turbulence is the normalised local velocity variance of the flow field:

    T_block = 1 - |mean(v)|² / mean(|v|²)      over the moving pixels of a block

T is 0 when every pixel in a block moves the same way (an orderly stream) and approaches 1
when directions cancel out (counter-flow, pushing, swirling). Scoring blocks locally means
two orderly streams side by side do not count as turbulent, mirroring the *local* velocity
variance in Helbing et al. (2007). Pixels slower than `min_speed` (background, people
standing still) are ignored, so a static crowd scores 0.

Why not the standard deviation of flow angles? Angles wrap around at ±180°: an orderly
crowd walking left has angles near +180° and -180°, so their standard deviation is ~180°
and it would score as maximum turbulence. The variance above works on the flow vectors
themselves, so it has no wrap-around problem.

Resolution: frames are downscaled to FLOW_MAX_WIDTH (default 320 px) before Farneback, which
is the CPU bottleneck. Flow vectors are scaled back to full-resolution pixels, so `min_speed`
means the same at any flow width, and the Farneback window and the turbulence block are
given for a 640 px reference width and scaled with the flow width, so they always cover the
same part of the scene. results/flow_speed_vs_accuracy.md compares the widths.

Camera motion: a panning, zooming or drifting camera (drones, handheld news footage) moves the
whole image, and background pixels that should count as static join the crowd's blocks with
the camera's motion. For each frame pair an affine model of the whole-frame motion is fitted to
the flow with RANSAC and subtracted, leaving people's motion relative to the ground. It is
applied only when the model explains at least half the frame and moves it by at least
`moving_camera_speed` (1 px/frame at 640 px wide); such a burst is marked "moving camera".
Slower whole-frame motion is left alone: a dense crowd walking one way across most of the frame
looks exactly like a slow pan (a synthetic fixed-camera clip produced 0.6-0.9 px/frame that
way), so fixed cameras score as before. Zones known to be fixed can switch the correction off
(`fixed_camera` in zones.json).

Density weighting (optional, TURBULENCE_WEIGHTING=density): each flow vector is weighted by the
CSRNet density at its position, so motion where the density map shows no people (water,
flags, sky, vehicles) adds nothing and blocks count in proportion to the people in them. With
no density map, or a uniform one, the score equals the whole-frame score.

Scene cuts: a frame pair is a cut when its colour histogram jumps (Bhattacharyya distance of
the hue-saturation histograms >= `cut_hist`) or its grey levels jump far more than the burst's
other pairs. A burst containing a cut is a "scene cut": the live pipeline and video analysis
skip it rather than compute flow across two shots.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import cv2
import numpy as np


REFERENCE_WIDTH = 640  # `winsize` and `block` are given for flow computed at this width


def scaled_to_width(size: int, width: int, minimum: int) -> int:
    """A size given for the reference width, scaled to a flow `width` in pixels."""
    return max(minimum, round(size * width / REFERENCE_WIDTH))


@dataclass(frozen=True)
class FlowParams:
    max_width: int = 320  # frames are downscaled to this width before computing flow (FLOW_MAX_WIDTH)
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15  # at the reference width; scaled with the flow width
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    # A frame pair is a cut when its colour histogram jumps by at least `cut_hist` (Bhattacharyya
    # distance; consecutive frames of one shot stay below ~0.25, cuts measured 0.4-0.75 on news
    # footage), or its mean grey-level change is at least `cut_min_diff` and more than
    # `cut_ratio` times the median change of the burst's other pairs (same-looking shots).
    cut_hist: float = 0.3
    cut_min_diff: float = 10.0
    cut_ratio: float = 4.0
    # Camera motion (see the module docstring).
    compensate_camera: bool = True
    moving_camera_speed: float = 1.0  # px/frame at the reference width: remove camera motion from this speed
    camera_min_inliers: float = 0.5  # share of the frame the camera model must explain
    camera_ransac_px: float = 0.5  # RANSAC inlier distance, flow-resolution px


@dataclass(frozen=True)
class TurbulenceParams:
    block: int = 32  # block size at the reference width; must be well above the Farneback window
    min_speed: float = 0.5  # full-resolution px/frame; slower pixels count as static
    min_moving_fraction: float = 0.2  # blocks with fewer moving pixels are skipped


@dataclass(frozen=True)
class TurbulenceResult:
    score: float  # 0..1
    mean_speed: float  # full-resolution px/frame, over moving pixels
    moving_fraction: float  # share of (ROI) pixels that are moving
    valid_blocks: int  # blocks that contributed to the score, summed over frame pairs
    pairs_used: int  # frame pairs used (cuts are skipped)
    camera_motion: float = 0.0  # estimated camera motion, full-resolution px/frame (median over pairs)
    moving_camera: bool = False  # it reached moving_camera_speed and was removed from every pair
    scene_cut: bool = False  # the burst spans a cut


@dataclass(frozen=True)
class CameraMotion:
    speed: float  # mean displacement of the fitted model over the frame, full-resolution px/frame
    inliers: float  # share of the frame's grid points the model explains
    displacement: np.ndarray | None  # (h, w, 2) the model's motion, full-resolution px/frame


def prepare_gray(frame: np.ndarray, max_width: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape
    if w > max_width:
        size = (max_width, max(1, round(h * max_width / w)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return gray


def _histogram(small: np.ndarray) -> np.ndarray:
    """Hue-saturation histogram of a BGR thumbnail (grey levels for a grey frame), normalised."""
    if small.ndim == 3:
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
    else:
        hist = cv2.calcHist([small], [0], None, [64], [0, 256])
    return cv2.normalize(hist, hist).flatten()


def find_cuts(frames: Sequence[np.ndarray], params: FlowParams = FlowParams()) -> list[bool]:
    """For each consecutive pair of BGR (or grey) frames: is it a cut rather than motion?"""
    h, w = frames[0].shape[:2]
    size = (160, max(1, round(h * 160 / w)))
    small = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
    grey = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in small]
    diffs = [float(cv2.absdiff(a, b).mean()) for a, b in zip(grey, grey[1:])]
    hists = [_histogram(f) for f in small]
    jumps = [float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)) for a, b in zip(hists, hists[1:])]
    cuts = []
    for i, (d, jump) in enumerate(zip(diffs, jumps)):
        others = diffs[:i] + diffs[i + 1 :]
        grey_cut = bool(others) and d >= params.cut_min_diff and d > params.cut_ratio * float(np.median(others))
        cuts.append(grey_cut or jump >= params.cut_hist)
    return cuts


def burst_has_cut(frames: Sequence[np.ndarray], params: FlowParams = FlowParams()) -> bool:
    """Does the burst span a shot change? (cheap: run it before CSRNet and flow)"""
    return any(find_cuts(frames, params))


def fit_camera_motion(flow: np.ndarray, full_width: int, params: FlowParams = FlowParams()) -> CameraMotion:
    """Affine model of the whole-frame motion in a flow (full-resolution px/frame, as
    compute_flows returns), fitted with RANSAC on a grid of flow vectors."""
    h, w = flow.shape[:2]
    to_full = full_width / w
    step = max(4, w // 40)
    ys, xs = np.mgrid[step // 2 : h : step, step // 2 : w : step]
    src = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    dst = src + flow[ys.ravel(), xs.ravel()] / to_full
    model, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=params.camera_ransac_px,
                                          maxIters=2000, confidence=0.99)
    if model is None:
        return CameraMotion(0.0, 0.0, None)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    du = (model[0, 0] - 1) * gx + model[0, 1] * gy + model[0, 2]
    dv = model[1, 0] * gx + (model[1, 1] - 1) * gy + model[1, 2]
    return CameraMotion(float(np.hypot(du, dv).mean()) * to_full, float(inliers.mean()),
                        (np.dstack([du, dv]) * to_full).astype(np.float32))


def compute_flows(frames: Sequence[np.ndarray], params: FlowParams = FlowParams()) -> list[np.ndarray]:
    """Dense flow between each pair of consecutive frames, skipping cuts.

    Each flow is an (h, w, 2) array at the flow resolution (at most `max_width` wide) whose
    vectors are in full-resolution pixels per frame.
    """
    grays = [prepare_gray(frame, params.max_width) for frame in frames]
    width = grays[0].shape[1]
    to_full = frames[0].shape[1] / width
    winsize = scaled_to_width(params.winsize, width, minimum=5)
    flows = []
    for prev, cur, cut in zip(grays, grays[1:], find_cuts(frames, params)):
        if cut:
            continue
        flow = cv2.calcOpticalFlowFarneback(
            prev, cur, None, params.pyr_scale, params.levels, winsize,
            params.iterations, params.poly_n, params.poly_sigma, 0,
        )
        flows.append(flow * to_full if to_full != 1 else flow)
    return flows


def polygon_mask(polygon: Sequence[tuple[float, float]], shape: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """Boolean mask of `shape` (h, w) inside `polygon`, given in pixels before `scale`."""
    mask = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(polygon, np.float64) * scale).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


@dataclass(frozen=True)
class _PairStats:
    weighted_turbulence: float  # sum over valid blocks of T_block * block weight
    weight: float  # block weights summed: moving pixels, or with density weighting, their density
    valid_blocks: int
    moving_pixels: int
    speed_sum: float
    roi_pixels: int


def _pair_stats(flow: np.ndarray, mask: np.ndarray | None, params: TurbulenceParams, b: int,
                weights: np.ndarray | None = None) -> _PairStats:
    h, w = flow.shape[:2]
    u = flow[..., 0].astype(np.float64)
    v = flow[..., 1].astype(np.float64)
    speed2 = u * u + v * v
    roi = np.ones((h, w), bool) if mask is None else mask
    moving = roi & (speed2 >= params.min_speed**2)
    moving_pixels = int(moving.sum())
    speed_sum = float(np.sqrt(speed2[moving]).sum())
    roi_pixels = int(roi.sum())

    hb, wb = h // b, w // b
    if hb == 0 or wb == 0:
        return _PairStats(0.0, 0, 0, moving_pixels, speed_sum, roi_pixels)

    def block_sum(a: np.ndarray) -> np.ndarray:
        return a[: hb * b, : wb * b].reshape(hb, b, wb, b).sum(axis=(1, 3))

    m = moving.astype(np.float64)
    valid = block_sum(m) >= params.min_moving_fraction * b * b  # enough moving pixels to judge the block
    w = m if weights is None else m * weights  # each moving pixel weighted by its density
    n = block_sum(w)
    valid &= n > 0
    if not valid.any():
        return _PairStats(0.0, 0, 0, moving_pixels, speed_sum, roi_pixels)

    n = n[valid]
    mean_u = block_sum(u * w)[valid] / n
    mean_v = block_sum(v * w)[valid] / n
    mean_speed2 = block_sum(speed2 * w)[valid] / n
    ratio = np.divide(mean_u**2 + mean_v**2, mean_speed2, out=np.ones_like(n), where=mean_speed2 > 0)
    turbulence = np.clip(1.0 - ratio, 0.0, 1.0)
    return _PairStats(
        float((turbulence * n).sum()), float(n.sum()), int(valid.sum()), moving_pixels, speed_sum, roi_pixels
    )


def density_weights(density: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A density map (any resolution covering the frame) as per-pixel weights at flow resolution."""
    h, w = shape
    resized = cv2.resize(np.clip(density.astype(np.float32), 0.0, None), (w, h), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float64)


def turbulence_from_flows(
    flows: Sequence[np.ndarray],
    mask: np.ndarray | None = None,
    params: TurbulenceParams = TurbulenceParams(),
    block: int | None = None,
    weights: np.ndarray | None = None,
) -> TurbulenceResult:
    """Turbulence of flows in full-resolution px/frame; `block` in flow pixels (default params.block).
    `weights` (h, w) at flow resolution weights every pixel, e.g. by density (see density_weights)."""
    stats = [_pair_stats(flow, mask, params, block or params.block, weights) for flow in flows]
    weight = sum(s.weight for s in stats)
    moving = sum(s.moving_pixels for s in stats)
    roi_pixels = sum(s.roi_pixels for s in stats)
    return TurbulenceResult(
        score=sum(s.weighted_turbulence for s in stats) / weight if weight else 0.0,
        mean_speed=sum(s.speed_sum for s in stats) / moving if moving else 0.0,
        moving_fraction=moving / roi_pixels if roi_pixels else 0.0,
        valid_blocks=sum(s.valid_blocks for s in stats),
        pairs_used=len(stats),
    )


@dataclass(frozen=True)
class _BurstFlows:
    raw: list[np.ndarray]  # Farneback flows of the burst's pairs (cuts skipped)
    flows: list[np.ndarray]  # the same with the camera's motion removed, if the camera moved
    camera_motion: float  # median camera-motion estimate over the pairs, full-resolution px/frame
    moving_camera: bool
    mask: np.ndarray | None  # ROI (and, for a moving camera, the inner area) at flow resolution
    block: int  # turbulence block at flow resolution


def _burst_flows(
    frames: Sequence[np.ndarray], roi: Sequence[tuple[float, float]] | None, params: FlowParams, block: int
) -> _BurstFlows:
    """Flows of a burst, with the camera's motion removed from every pair when the burst's median
    camera motion reaches `moving_camera_speed` (decided once per burst, so a burst is either
    corrected and marked "moving camera" or left exactly as it was)."""
    raw = compute_flows(frames, params)
    full_width = frames[0].shape[1]
    fits = [fit_camera_motion(flow, full_width, params) for flow in raw] if params.compensate_camera else []
    speeds = [fit.speed if fit.inliers >= params.camera_min_inliers else 0.0 for fit in fits]
    camera = float(np.median(speeds)) if speeds else 0.0
    moving = bool(fits) and camera >= params.moving_camera_speed * full_width / REFERENCE_WIDTH
    flows = [flow - fit.displacement if fit.displacement is not None else flow for flow, fit in zip(raw, fits)] \
        if moving else raw
    block = scaled_to_width(block, min(full_width, params.max_width), minimum=4)
    mask = None
    if roi is not None and raw:
        fh, fw = raw[0].shape[:2]
        mask = polygon_mask(roi, (fh, fw), scale=fw / full_width)
    if moving:
        # With the camera moving, content enters and leaves the view at the border, where flow
        # has no true match: leave the outer block out of the score.
        fh, fw = raw[0].shape[:2]
        inner = np.zeros((fh, fw), bool)
        inner[block:-block, block:-block] = True
        mask = inner if mask is None else mask & inner
    return _BurstFlows(raw, flows, camera, moving, mask, block)


def burst_turbulence(
    frames: Sequence[np.ndarray],
    roi: Sequence[tuple[float, float]] | None = None,
    flow_params: FlowParams = FlowParams(),
    turbulence_params: TurbulenceParams = TurbulenceParams(),
    density: np.ndarray | None = None,
) -> TurbulenceResult:
    """Turbulence of a burst of consecutive BGR frames, optionally inside an ROI polygon, with
    the camera's own motion removed (see the module docstring). Pass the burst's CSRNet
    `density` map to weight the flow by where the people are."""
    b = _burst_flows(frames, roi, flow_params, turbulence_params.block)
    weights = density_weights(density, b.raw[0].shape[:2]) if density is not None and b.raw else None
    result = turbulence_from_flows(b.flows, b.mask, turbulence_params, b.block, weights)
    return replace(result, camera_motion=b.camera_motion, moving_camera=b.moving_camera,
                   scene_cut=len(b.raw) < len(frames) - 1)


def compute_flow(
    frames: Sequence[np.ndarray],
    roi: Sequence[tuple[float, float]] | None = None,
    flow_params: FlowParams = FlowParams(),
    turbulence_params: TurbulenceParams = TurbulenceParams(),
) -> dict:
    """Flow summary for a burst of consecutive BGR frames (5 frames give 4 flow maps).

    Keys: 'magnitude' (mean raw flow magnitude over all pixels and pairs, full-resolution
    px/frame), 'turbulence' (0..1, camera motion removed), 'flow_maps' (the raw (h, w, 2)
    flows at flow resolution, in full-resolution px/frame), 'camera_motion' (estimated px/frame),
    and the diagnostics 'moving_fraction' and 'pairs_used'.
    """
    b = _burst_flows(frames, roi, flow_params, turbulence_params.block)
    flows, mask = b.raw, b.mask
    result = turbulence_from_flows(b.flows, mask, turbulence_params, b.block)
    magnitudes = [np.linalg.norm(f, axis=2) for f in flows]
    magnitude = float(np.mean([m[mask].mean() if mask is not None else m.mean() for m in magnitudes])) if flows else 0.0
    return {
        "magnitude": magnitude,
        "turbulence": result.score,
        "flow_maps": flows,
        "camera_motion": b.camera_motion,
        "moving_fraction": result.moving_fraction,
        "pairs_used": result.pairs_used,
    }


def turbulence_to_label(score: float) -> str:
    """LOW below 0.3, MEDIUM below 0.6, HIGH from 0.6."""
    if score >= 0.6:
        return "HIGH"
    if score >= 0.3:
        return "MEDIUM"
    return "LOW"


def visualise_flow(frame: np.ndarray, flow_map: np.ndarray, alpha: float = 0.6, max_speed: float | None = None) -> np.ndarray:
    """Colour-coded flow over the frame: hue = direction, brightness = speed (BGR, frame size).

    Brightness is scaled to the fastest pixel (at least 1 px/frame, so a still scene stays dark);
    pass `max_speed` to compare frames on one scale.
    """
    h, w = frame.shape[:2]
    magnitude, angle = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1])  # angle in [0, 2π)
    hsv = np.zeros((*flow_map.shape[:2], 3), np.uint8)
    hsv[..., 0] = (angle * 90 / np.pi).astype(np.uint8)  # OpenCV hue runs 0..179
    hsv[..., 1] = 255
    scale = max_speed or max(float(magnitude.max()), 1.0)
    hsv[..., 2] = np.clip(magnitude / scale * 255, 0, 255).astype(np.uint8)
    colour = cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), (w, h), interpolation=cv2.INTER_LINEAR)
    base = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 1 - alpha, colour, alpha, 0)


def visualise_vectors(frame: np.ndarray, flow: np.ndarray, step: int = 16, scale: float = 3.0) -> np.ndarray:
    """Draw flow arrows (scaled by `scale`) every `step` pixels, for debugging and figures."""
    h, w = flow.shape[:2]
    canvas = cv2.resize(frame, (w, h)) if frame.shape[:2] != (h, w) else frame.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = flow[y, x]
            if dx * dx + dy * dy < 0.25:
                continue
            tip = (int(round(x + dx * scale)), int(round(y + dy * scale)))
            cv2.arrowedLine(canvas, (x, y), tip, (0, 255, 0), 1, tipLength=0.3)
    return canvas
