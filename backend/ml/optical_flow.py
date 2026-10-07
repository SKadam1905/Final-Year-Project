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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowParams:
    max_width: int = 640  # frames are downscaled to this width before computing flow
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    # A frame pair is a cut (e.g. a looping demo clip jumping back to its start) and is
    # skipped when its mean grey-level change is at least `cut_min_diff` and more than
    # `cut_ratio` times the median change of the burst's other pairs.
    cut_min_diff: float = 10.0
    cut_ratio: float = 4.0


@dataclass(frozen=True)
class TurbulenceParams:
    block: int = 32  # block size in flow pixels; must be well above the Farneback window
    min_speed: float = 0.5  # px/frame at flow resolution; slower pixels count as static
    min_moving_fraction: float = 0.2  # blocks with fewer moving pixels are skipped


@dataclass(frozen=True)
class TurbulenceResult:
    score: float  # 0..1
    mean_speed: float  # px/frame at flow resolution, over moving pixels
    moving_fraction: float  # share of (ROI) pixels that are moving
    valid_blocks: int  # blocks that contributed to the score, summed over frame pairs
    pairs_used: int  # frame pairs used (cuts are skipped)


def prepare_gray(frame: np.ndarray, max_width: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape
    if w > max_width:
        size = (max_width, max(1, round(h * max_width / w)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return gray


def find_cuts(grays: Sequence[np.ndarray], params: FlowParams = FlowParams()) -> list[bool]:
    """For each consecutive pair: is it a cut rather than motion? (needs >= 3 frames)"""
    h, w = grays[0].shape
    size = (160, max(1, round(h * 160 / w)))
    small = [cv2.resize(g, size, interpolation=cv2.INTER_AREA) for g in grays]
    diffs = [float(cv2.absdiff(a, b).mean()) for a, b in zip(small, small[1:])]
    cuts = []
    for i, d in enumerate(diffs):
        others = diffs[:i] + diffs[i + 1 :]
        cuts.append(bool(others) and d >= params.cut_min_diff and d > params.cut_ratio * float(np.median(others)))
    return cuts


def compute_flows(frames: Sequence[np.ndarray], params: FlowParams = FlowParams()) -> list[np.ndarray]:
    """Dense flow (h, w, 2) between each pair of consecutive frames, skipping cuts."""
    grays = [prepare_gray(frame, params.max_width) for frame in frames]
    flows = []
    for prev, cur, cut in zip(grays, grays[1:], find_cuts(grays, params)):
        if cut:
            continue
        flows.append(
            cv2.calcOpticalFlowFarneback(
                prev, cur, None, params.pyr_scale, params.levels, params.winsize,
                params.iterations, params.poly_n, params.poly_sigma, 0,
            )
        )
    return flows


def polygon_mask(polygon: Sequence[tuple[float, float]], shape: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """Boolean mask of `shape` (h, w) inside `polygon`, given in pixels before `scale`."""
    mask = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(polygon, np.float64) * scale).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


@dataclass(frozen=True)
class _PairStats:
    weighted_turbulence: float  # sum over valid blocks of T_block * moving pixels
    weight: int  # moving pixels in valid blocks
    valid_blocks: int
    moving_pixels: int
    speed_sum: float
    roi_pixels: int


def _pair_stats(flow: np.ndarray, mask: np.ndarray | None, params: TurbulenceParams) -> _PairStats:
    b = params.block
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
    n = block_sum(m)
    valid = n >= params.min_moving_fraction * b * b
    if not valid.any():
        return _PairStats(0.0, 0, 0, moving_pixels, speed_sum, roi_pixels)

    n = n[valid]
    mean_u = block_sum(u * m)[valid] / n
    mean_v = block_sum(v * m)[valid] / n
    mean_speed2 = block_sum(speed2 * m)[valid] / n
    ratio = np.divide(mean_u**2 + mean_v**2, mean_speed2, out=np.ones_like(n), where=mean_speed2 > 0)
    turbulence = np.clip(1.0 - ratio, 0.0, 1.0)
    return _PairStats(
        float((turbulence * n).sum()), int(n.sum()), int(valid.sum()), moving_pixels, speed_sum, roi_pixels
    )


def turbulence_from_flows(
    flows: Sequence[np.ndarray], mask: np.ndarray | None = None, params: TurbulenceParams = TurbulenceParams()
) -> TurbulenceResult:
    stats = [_pair_stats(flow, mask, params) for flow in flows]
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


def _burst_flows(
    frames: Sequence[np.ndarray], roi: Sequence[tuple[float, float]] | None, params: FlowParams
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Flows of a burst plus the ROI mask at flow resolution (None = whole frame)."""
    flows = compute_flows(frames, params)
    mask = None
    if roi is not None and flows:
        fh, fw = flows[0].shape[:2]
        mask = polygon_mask(roi, (fh, fw), scale=fw / frames[0].shape[1])
    return flows, mask


def burst_turbulence(
    frames: Sequence[np.ndarray],
    roi: Sequence[tuple[float, float]] | None = None,
    flow_params: FlowParams = FlowParams(),
    turbulence_params: TurbulenceParams = TurbulenceParams(),
) -> TurbulenceResult:
    """Turbulence of a burst of consecutive BGR frames, optionally inside an ROI polygon."""
    flows, mask = _burst_flows(frames, roi, flow_params)
    return turbulence_from_flows(flows, mask, turbulence_params)


def compute_flow(
    frames: Sequence[np.ndarray],
    roi: Sequence[tuple[float, float]] | None = None,
    flow_params: FlowParams = FlowParams(),
    turbulence_params: TurbulenceParams = TurbulenceParams(),
) -> dict:
    """Flow summary for a burst of consecutive BGR frames (5 frames give 4 flow maps).

    Keys: 'magnitude' (mean flow magnitude over all pixels and pairs, px/frame at flow
    resolution), 'turbulence' (0..1), 'flow_maps' (the raw (h, w, 2) flows), and the
    diagnostics 'moving_fraction' and 'pairs_used'.
    """
    flows, mask = _burst_flows(frames, roi, flow_params)
    result = turbulence_from_flows(flows, mask, turbulence_params)
    magnitudes = [np.linalg.norm(f, axis=2) for f in flows]
    magnitude = float(np.mean([m[mask].mean() if mask is not None else m.mean() for m in magnitudes])) if flows else 0.0
    return {
        "magnitude": magnitude,
        "turbulence": result.score,
        "flow_maps": flows,
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
