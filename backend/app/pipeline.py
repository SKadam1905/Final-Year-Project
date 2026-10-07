"""Per-zone analysis of one frame burst: CSRNet density + optical-flow turbulence -> risk.

Used by tools/analyze_video.py now, and by the AI worker in Phase 3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.calibration import ZoneGeometry
from app.config import Settings, ZoneConfig, get_settings
from app.csrnet_model import DensityEstimator, DensityResult
from app.risk_engine import RiskEngine, RiskThresholds, ZoneStatus
from ml.optical_flow import FlowParams, TurbulenceParams, TurbulenceResult, burst_turbulence


@dataclass(frozen=True)
class BurstAnalysis:
    status: ZoneStatus
    density: DensityResult
    turbulence: TurbulenceResult
    inference_ms: float
    flow_ms: float


class ZoneAnalyzer:
    def __init__(
        self,
        zone: ZoneConfig,
        estimator: DensityEstimator,
        settings: Settings | None = None,
        flow_params: FlowParams = FlowParams(),
        turbulence_params: TurbulenceParams = TurbulenceParams(),
    ) -> None:
        s = settings or get_settings()
        self.zone = zone
        self.estimator = estimator
        self.geometry = ZoneGeometry.from_zone(zone)
        self.engine = RiskEngine(
            zone.id,
            window=s.rolling_maxlen,
            thresholds=RiskThresholds(s.risk_yellow, s.risk_orange, s.risk_red),
            k=s.pressure_k,
            window_s=s.rolling_window_s,
        )
        self.flow_params = flow_params
        self.turbulence_params = turbulence_params

    def analyze(self, frames: Sequence[np.ndarray], timestamp: float) -> BurstAnalysis:
        """CSRNet on the middle frame (frame 3 of 5), Farneback flow across the burst."""
        started = time.perf_counter()
        density = self.estimator.predict(frames[len(frames) // 2])
        inferred = time.perf_counter()
        count, per_m2 = self.geometry.measure(density.density, density.frame_size)
        turbulence = burst_turbulence(frames, self.zone.roi, self.flow_params, self.turbulence_params)
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
        )
        return BurstAnalysis(status, density, turbulence, (inferred - started) * 1e3, (done - inferred) * 1e3)
