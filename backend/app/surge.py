"""Speed surge: a fast trigger next to the smoothed crowd pressure (optional; results/umn_speed_surge.md).

    surge = speed / median speed of the zone's readings in the previous SURGE_HISTORY_S seconds

`speed` is the burst's mean speed of moving pixels, which the turbulence step already computes
(ml.optical_flow.TurbulenceResult.mean_speed), so the surge needs no extra optical flow. The median is
floored at the "moving" threshold (TurbulenceParams.min_speed, 0.5 px/frame), so an empty, still scene
cannot divide by ~0. Being a ratio to the same camera's recent pace, it needs no per-camera calibration.

The rule fires while the last SURGE_CONSECUTIVE readings all reach SURGE_RATIO. With SURGE_TRIGGER=1,
app.pipeline raises a zone that is already at SURGE_MIN_LEVEL or above to at least ORANGE while it fires
(reason "speed surge"); the alert policy (cooldown, escalation) is the usual one. The smoothed pressure
itself is never changed: on the UMN dataset averaging the surge into the 30 s mean caught 4 of 11 panics,
two readings in a row at 1.10 caught 9 of 11 (leave-one-scene-out), at ~0.7 false alarms per minute of
normal walking before the density gate. tools/surge_false_alarms.py measures the rate on other footage.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass

from ml.optical_flow import TurbulenceParams

MIN_SPEED = TurbulenceParams().min_speed
SURGE_REASON = "speed surge"


@dataclass(frozen=True)
class SurgeReading:
    speed: float  # this reading's mean speed of moving pixels, full-resolution px/frame
    ratio: float | None  # speed / recent median; None until the zone has history
    streak: int  # readings in a row at or above the ratio threshold, this one included
    fired: bool  # streak >= the required number of consecutive readings


class SurgeTracker:
    """Surge ratio and rule state for one zone; feed it one reading per burst."""

    def __init__(self, ratio: float = 1.10, consecutive: int = 2, history_s: float = 10.0,
                 min_speed: float = MIN_SPEED) -> None:
        if ratio <= 0 or consecutive < 1 or history_s <= 0:
            raise ValueError("SURGE_RATIO and SURGE_HISTORY_S must be > 0 and SURGE_CONSECUTIVE >= 1")
        self.ratio = ratio
        self.consecutive = consecutive
        self.history_s = history_s
        self.min_speed = min_speed
        self.reset()

    def reset(self) -> None:
        """Forget the history, e.g. after a scene cut, when the speeds before it are another view's."""
        self._history: deque[tuple[float, float]] = deque()
        self._streak = 0

    def update(self, speed: float, timestamp: float) -> SurgeReading:
        while self._history and timestamp - self._history[0][0] > self.history_s:
            self._history.popleft()
        previous = [s for _, s in self._history]
        ratio = speed / max(statistics.median(previous), self.min_speed) if previous else None
        self._streak = self._streak + 1 if ratio is not None and ratio >= self.ratio else 0
        self._history.append((timestamp, speed))
        return SurgeReading(speed, ratio, self._streak, self._streak >= self.consecutive)
