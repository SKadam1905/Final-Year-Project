"""tools/surge_false_alarms: speed-surge firings per hour on a video, read like a live camera."""

import numpy as np

from ml.optical_flow import FlowParams
from tests.conftest import texture, write_video
from tools.surge_false_alarms import bursts, measure


def moving_video(path, speeds, seconds_each, fps=10):
    """A texture sliding right at `speeds` px/frame, each for `seconds_each` seconds."""
    base = np.dstack([texture(64, 400, seed=3)] * 3)
    frames, x = [], 0.0
    for speed in speeds:
        for _ in range(int(seconds_each * fps)):
            frames.append(np.ascontiguousarray(np.roll(base, int(round(x)), axis=1)[:, :96]))
            x += speed
    return write_video(path, frames, fps=fps)


def test_bursts_are_taken_like_a_live_camera(tmp_path):
    video = moving_video(tmp_path / "v.avi", [1.0], 6, fps=10)  # 60 frames
    taken = list(bursts(video, interval_s=2.0, size=5))
    assert [t for t, _ in taken] == [0.0, 2.0, 4.0] and all(len(f) == 5 for _, f in taken)


def test_steady_walking_gives_no_alarm_and_a_surge_does(tmp_path):
    flow = FlowParams(max_width=320, compensate_camera=False)  # the whole frame moves: no camera to remove here
    steady = moving_video(tmp_path / "steady.avi", [1.5], 40)
    rows, info = measure(steady, [1.5], consecutive=2, history_s=10, interval_s=2, burst_size=5, flow=flow)
    assert info["readings"] == 20 and rows[0]["false alarms (switch-ons)"] == 0
    surge = moving_video(tmp_path / "surge.avi", [1.5, 4.5, 1.5], 14)  # 14 s steady, 14 s three times faster
    rows, info = measure(surge, [1.5, 5.0], consecutive=2, history_s=10, interval_s=2, burst_size=5, flow=flow)
    by_ratio = {r["ratio"]: r for r in rows}
    assert by_ratio[1.5]["false alarms (switch-ons)"] == 1  # one surge -> fires once (for several readings)
    assert by_ratio[5.0]["false alarms (switch-ons)"] == 0  # 3x faster never reaches 5x
    assert info["hours"] > 0 and by_ratio[1.5]["per hour, rule alone"] != "-"
