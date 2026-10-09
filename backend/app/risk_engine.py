"""Crowd-pressure risk score, risk levels and per-zone smoothing (plan §4.3, §4.4, §4.6).

The risk score is

    pressure = density * (1 + k * turbulence)

* density    - persons per m² in the zone (CSRNet count / calibrated ground area)
* turbulence - normalised local velocity variance from optical flow, 0..1
* k          - how strongly turbulence amplifies density (default 0.5; tune on clips)

This is *inspired by* Helbing, Johansson & Al-Abideen (2007), Phys. Rev. E 75, 046109,
whose crowd pressure is density x local velocity variance in physical units (m/s); it is
not their formula. With k = 0.5 a fully turbulent crowd scores 1.5x its density, so the
thresholds stay in "persons/m² equivalent" units.

Alerts use the mean of the last 15 scores (30 s at one burst every 2 s), which suppresses
one-frame spikes at the cost of up to ~30 s of extra delay. Report that delay separately
from alert delivery time (plan §4.6).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from enum import StrEnum


class RiskLevel(StrEnum):
    """Risk levels as plain strings ("RED" == RiskLevel.RED), ordered by severity."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"

    @property
    def severity(self) -> int:
        return _SEVERITY[self]

    def __lt__(self, other: str) -> bool:
        return self.severity < RiskLevel(other).severity

    def __le__(self, other: str) -> bool:
        return self.severity <= RiskLevel(other).severity

    def __gt__(self, other: str) -> bool:
        return self.severity > RiskLevel(other).severity

    def __ge__(self, other: str) -> bool:
        return self.severity >= RiskLevel(other).severity


_SEVERITY = {level: rank for rank, level in enumerate(RiskLevel)}


@dataclass(frozen=True)
class RiskThresholds:
    """Lower bounds of YELLOW, ORANGE and RED on the pressure scale."""

    yellow: float = 2.0
    orange: float = 4.0
    red: float = 6.0

    def __post_init__(self) -> None:
        if not 0 <= self.yellow < self.orange < self.red:
            raise ValueError(f"thresholds must satisfy 0 <= yellow < orange < red, got {self}")


def crowd_pressure(density: float, turbulence: float, k: float = 0.5) -> float:
    if density < 0 or math.isnan(density):
        raise ValueError(f"density must be a non-negative number, got {density}")
    turbulence = min(max(turbulence, 0.0), 1.0)
    return density * (1.0 + k * turbulence)


def classify_risk(pressure: float, thresholds: RiskThresholds = RiskThresholds()) -> RiskLevel:
    if pressure >= thresholds.red:
        return RiskLevel.RED
    if pressure >= thresholds.orange:
        return RiskLevel.ORANGE
    if pressure >= thresholds.yellow:
        return RiskLevel.YELLOW
    return RiskLevel.GREEN


class RollingWindow:
    """Mean of the samples from the last `window_s` seconds (at most `maxlen` of them).

    Expiring by time as well as by count means a camera outage does not leave stale
    readings in the average.
    """

    def __init__(self, window_s: float = 30.0, maxlen: int = 15) -> None:
        self.window_s = window_s
        self._samples: deque[tuple[float, float]] = deque(maxlen=maxlen)

    def add(self, timestamp: float, value: float) -> None:
        self._samples.append((timestamp, value))
        while self._samples and timestamp - self._samples[0][0] >= self.window_s:
            self._samples.popleft()

    def mean(self) -> float:
        if not self._samples:
            raise ValueError("no samples in window")
        return sum(value for _, value in self._samples) / len(self._samples)

    def __len__(self) -> int:
        return len(self._samples)


class RiskEngine:
    """Smoothed risk level for one camera zone: create one instance per zone.

    `update()` takes one reading per burst and returns a dict with zone_id, raw_pressure,
    smoothed_pressure (mean of the last `window` readings within `window_s` seconds),
    risk_level, density, turbulence and samples. The level is classified from the
    smoothed pressure.
    """

    def __init__(
        self,
        zone_id: str,
        window: int = 15,
        thresholds: RiskThresholds = RiskThresholds(),
        k: float = 0.5,
        window_s: float = 30.0,
    ) -> None:
        self.zone_id = zone_id
        self.window = window
        self.window_s = window_s
        self.thresholds = thresholds
        self.k = k
        self.reset()

    def update(self, density: float, turbulence: float, timestamp: float | None = None) -> dict:
        pressure = crowd_pressure(density, turbulence, self.k)
        self._readings.add(time.time() if timestamp is None else timestamp, pressure)
        smoothed = self._readings.mean()
        self._status = {
            "zone_id": self.zone_id,
            "raw_pressure": pressure,
            "smoothed_pressure": smoothed,
            "risk_level": classify_risk(smoothed, self.thresholds),
            "density": density,
            "turbulence": turbulence,
            "samples": len(self._readings),
        }
        return self._status

    def get_status(self) -> dict | None:
        """The last update's result, or None before the first reading."""
        return self._status

    def reset(self) -> None:
        self._readings = RollingWindow(self.window_s, self.window)
        self._status: dict | None = None


@dataclass(frozen=True)
class ZoneStatus:
    """One zone's reading as passed to alerts, storage and the API."""

    zone_id: str
    timestamp: float
    count: float
    density: float
    turbulence: float
    pressure: float
    smoothed_pressure: float
    level: RiskLevel
    samples: int
    count_method: str = "density"  # which model gave `count`: "density" (CSRNet) or "detection" (YOLO)
    density_count: float | None = None  # CSRNet's count
    detection_count: int | None = None  # YOLO's count (None without a detector)
    hotspot: dict | None = None  # the most crowded grid cell: row, col, label, count, density

    def to_dict(self) -> dict:
        data = asdict(self)
        data["level"] = str(self.level)
        return data
