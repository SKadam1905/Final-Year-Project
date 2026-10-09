"""Per-zone analysis of one frame burst: crowd count (CSRNet density, or YOLO detection in sparse
scenes) + optical-flow turbulence -> risk, plus a per-cell grid of the camera view.

Used by the live worker, video analysis and tools/analyze_video.py. A burst that spans a
scene cut is skipped before CSRNet runs: it gives no reading and does not enter the rolling
risk average.

Hybrid count (choose_count): a detector is accurate when people are few and fully visible, a
density map when they are many and hide each other. YOLO's count is used when the scene is
sparse, i.e. fewer than HYBRID_THRESHOLD people; otherwise CSRNet's. With HYBRID_RULE=agree (the
default) CSRNet must also count fewer than the threshold: YOLO misses people who are small or
hidden (it finds almost none in an aerial view of a dense crowd), so its count alone cannot say a
scene is sparse. HYBRID_RULE=detector lets YOLO's count decide alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.calibration import ZoneGeometry, ZoneGrid
from app.config import Settings, ZoneConfig, get_settings
from app.csrnet_model import DensityEstimator, DensityResult
from app.risk_engine import RiskEngine, RiskThresholds, ZoneStatus
from ml.optical_flow import FlowParams, TurbulenceParams, TurbulenceResult, burst_has_cut, burst_turbulence

HYBRID_RULES = ("agree", "detector")


@dataclass(frozen=True)
class BurstAnalysis:
    status: ZoneStatus | None  # None for a scene cut
    density: DensityResult | None
    turbulence: TurbulenceResult | None
    inference_ms: float
    flow_ms: float
    scene_cut: bool = False
    detections: object | None = None  # app.detector.Detections (all of them, inside the ROI or not)
    grid: ZoneGrid | None = None
    detection_ms: float = 0.0


def choose_count(density_count: float, detection_count: int | None, threshold: float, rule: str = "agree") -> str:
    """"detection" when the scene is sparse (fewer than `threshold` people), else "density"."""
    if detection_count is None or detection_count >= threshold:
        return "density"
    if rule == "agree" and density_count >= threshold:
        return "density"  # YOLO sees few people but CSRNet many: small or hidden people, a dense crowd
    return "detection"


class ZoneAnalyzer:
    def __init__(
        self,
        zone: ZoneConfig,
        estimator: DensityEstimator,
        settings: Settings | None = None,
        flow_params: FlowParams | None = None,
        turbulence_params: TurbulenceParams = TurbulenceParams(),
        detector=None,  # app.detector.PersonDetector, or None for CSRNet counts only
    ) -> None:
        s = settings or get_settings()
        self.zone = zone
        self.estimator = estimator
        self.detector = detector
        self.geometry = ZoneGeometry.from_zone(zone)
        self.engine = RiskEngine(
            zone.id,
            window=s.rolling_maxlen,
            thresholds=RiskThresholds(s.risk_yellow, s.risk_orange, s.risk_red),
            k=s.pressure_k,
            window_s=s.rolling_window_s,
        )
        # A zone marked fixed_camera never has whole-frame motion removed (see ml.optical_flow).
        self.flow_params = flow_params or FlowParams(max_width=s.flow_max_width,
                                                     compensate_camera=not zone.fixed_camera)
        self.turbulence_params = turbulence_params
        if s.turbulence_weighting not in ("frame", "density"):
            raise ValueError(f"TURBULENCE_WEIGHTING must be 'frame' or 'density', not {s.turbulence_weighting!r}")
        self.density_weighted = s.turbulence_weighting == "density"
        if s.hybrid_rule not in HYBRID_RULES:
            raise ValueError(f"HYBRID_RULE must be one of {HYBRID_RULES}, not {s.hybrid_rule!r}")
        self.hybrid_threshold = s.hybrid_threshold
        self.hybrid_rule = s.hybrid_rule
        self.grid_shape = (s.grid_rows, s.grid_cols)

    def analyze(self, frames: Sequence[np.ndarray], timestamp: float) -> BurstAnalysis:
        """CSRNet (and YOLO) on the middle frame (frame 3 of 5), Farneback flow across the burst. A
        burst spanning a scene cut returns scene_cut=True and leaves the risk engine untouched."""
        started = time.perf_counter()
        if burst_has_cut(frames, self.flow_params):
            return BurstAnalysis(None, None, None, 0.0, (time.perf_counter() - started) * 1e3, scene_cut=True)
        frame = frames[len(frames) // 2]
        density = self.estimator.predict(frame)
        inferred = time.perf_counter()
        detections = self.detector.detect(frame) if self.detector is not None else None
        detected = time.perf_counter()

        density_count, _ = self.geometry.measure(density.density, density.frame_size)
        detection_count = int(self.geometry.in_zone(detections.feet).sum()) if detections is not None else None
        method = choose_count(density_count, detection_count, self.hybrid_threshold, self.hybrid_rule)
        count = float(detection_count) if method == "detection" else density_count
        per_m2 = max(count, 0.0) / self.geometry.ground_area(density.frame_size, density.density.shape)
        grid = self.geometry.grid(density.density, density.frame_size, *self.grid_shape,
                                  points=detections.feet if method == "detection" else None)

        turbulence = burst_turbulence(frames, self.zone.roi, self.flow_params, self.turbulence_params,
                                      density.density if self.density_weighted else None)
        done = time.perf_counter()
        risk = self.engine.update(per_m2, turbulence.score, timestamp)
        status = ZoneStatus(
            zone_id=self.zone.id,
            timestamp=timestamp,
            count=count,
            density=per_m2,
            turbulence=turbulence.score,
            pressure=risk["raw_pressure"],
            smoothed_pressure=risk["smoothed_pressure"],
            level=risk["risk_level"],
            samples=risk["samples"],
            count_method=method,
            density_count=density_count,
            detection_count=detection_count,
            hotspot=grid.hotspot(),
        )
        return BurstAnalysis(status, density, turbulence, (inferred - started) * 1e3, (done - detected) * 1e3,
                             detections=detections, grid=grid, detection_ms=(detected - inferred) * 1e3)
