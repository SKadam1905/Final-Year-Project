"""Alarm metrics (ml.event_metrics) used to validate the turbulence score on UMN."""

import numpy as np
import pytest

from ml.event_metrics import alarm_before, detection_delay, false_alarms, segments, switch_ons


def test_switch_ons_count_rising_edges():
    assert switch_ons(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)) == 3
    assert switch_ons(np.array([1, 1, 0], bool)) == 1  # already on at the start counts once
    assert switch_ons(np.array([], bool)) == 0


def test_false_alarms_per_minute_within_runs_of_frames():
    frames = np.r_[np.arange(0, 900), np.arange(2000, 2900)]  # two 30 s stretches of normal frames at 30 fps
    scores = np.zeros(frames.size)
    scores[100:130] = 0.8  # one alarm in the first stretch
    scores[899] = 0.8  # the last frame of the first stretch...
    scores[900] = 0.8  # ...and the first of the second: two separate stretches, two switch-ons
    fa = false_alarms(frames, scores, 0.5, fps=30)
    assert fa.count == 3 and fa.minutes == pytest.approx(1.0) and fa.per_minute == pytest.approx(3.0)
    assert [(s.start, s.stop) for s in segments(frames)] == [(0, 900), (900, 1800)]


def test_detection_delay_and_alarm_before_onset():
    frames = np.arange(100, 200)
    scores = np.zeros(100)
    scores[frames >= 160] = 0.7  # alarm from frame 160
    assert detection_delay(frames, scores, onset=150, threshold=0.5, fps=30) == pytest.approx(10 / 30)
    assert detection_delay(frames, scores, onset=150, threshold=0.9, fps=30) is None
    assert detection_delay(frames, scores, onset=150, threshold=0.5, fps=30, last=155) is None
    scores[frames == 140] = 0.7
    assert alarm_before(frames, scores, onset=150, threshold=0.5, fps=30, seconds=1.0)  # 140 is within 30 frames
    assert not alarm_before(frames, scores, onset=150, threshold=0.5, fps=30, seconds=0.2)
