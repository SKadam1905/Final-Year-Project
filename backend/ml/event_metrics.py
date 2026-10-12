"""Alarm metrics for a per-frame score against labelled events (turbulence validation, Phase 6B).

An alarm is "score >= threshold". Detection delay: seconds from an event's onset to the first frame
at or after it whose score reaches the threshold (None: never within the event). False alarms: the
number of times the alarm switches on (a rising edge, or a segment that starts already on) inside
stretches of normal frames, per minute of normal footage. Counting switch-ons rather than frames
means one long alarm is one false alarm, as an operator would experience it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FalseAlarms:
    count: int
    minutes: float

    @property
    def per_minute(self) -> float:
        return self.count / self.minutes if self.minutes else float("nan")


def switch_ons(alarm: np.ndarray) -> int:
    """How often a boolean alarm series turns on (an alarm already on in the first frame counts)."""
    alarm = np.asarray(alarm, bool)
    if alarm.size == 0:
        return 0
    return int(alarm[0]) + int(np.count_nonzero(alarm[1:] & ~alarm[:-1]))


def segments(frames: np.ndarray) -> list[slice]:
    """Runs of consecutive frame numbers, as slices into `frames` (sorted)."""
    frames = np.asarray(frames)
    if frames.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    bounds = [0, *breaks.tolist(), frames.size]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def false_alarms(frames: np.ndarray, scores: np.ndarray, threshold: float, fps: float) -> FalseAlarms:
    """Switch-ons of score >= threshold within each run of consecutive (normal) frames."""
    frames, scores = np.asarray(frames), np.asarray(scores, float)
    count = sum(switch_ons(scores[s] >= threshold) for s in segments(frames))
    return FalseAlarms(count, frames.size / fps / 60.0)


def detection_delay(frames: np.ndarray, scores: np.ndarray, onset: int, threshold: float, fps: float,
                    last: int | None = None) -> float | None:
    """Seconds from `onset` to the first frame in [onset, last] with score >= threshold, or None."""
    frames, scores = np.asarray(frames), np.asarray(scores, float)
    window = (frames >= onset) & (frames <= (frames.max() if last is None else last))
    hits = frames[window & (scores >= threshold)]
    return float(hits.min() - onset) / fps if hits.size else None


def alarm_before(frames: np.ndarray, scores: np.ndarray, onset: int, threshold: float, fps: float,
                 seconds: float) -> bool:
    """Was the alarm on at any frame in the `seconds` just before `onset`?"""
    frames, scores = np.asarray(frames), np.asarray(scores, float)
    window = (frames < onset) & (frames >= onset - round(seconds * fps))
    return bool((scores[window] >= threshold).any())
