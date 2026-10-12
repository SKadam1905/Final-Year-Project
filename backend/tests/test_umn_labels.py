"""UMN ground truth from the video's caption (scripts/umn_labels.py) and per-frame labels (scripts/umn_validation.py)."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np

SCRIPTS = Path(__file__).parents[2] / "scripts"


def load(name, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))  # as when run as a script: report_utils sits beside it
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_caption_detection_intervals_and_clip_starts(monkeypatch):
    umn = load("umn_labels", monkeypatch)
    frame = np.full((240, 320, 3), 120, np.uint8)
    assert not umn.caption_present(frame)
    cv2.putText(frame, "Abnormal Crowd Activity", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
    assert umn.caption_present(frame)

    flags = np.zeros(40, bool)
    flags[5:9] = flags[20:31] = True
    assert umn.intervals(flags) == [(5, 8), (20, 30)]
    diffs = np.ones(40)
    diffs[12] = 50.0  # the hard cut 3 frames after the first caption ends
    assert umn.clip_starts(diffs, [8, 30], fps=5) == [0, 12]

    assert umn.scene_of(np.array([90, 150, 80]), saturation=90) == "lawn"
    assert umn.scene_of(np.array([150, 170, 190]), saturation=60) == "plaza"
    assert umn.scene_of(np.array([120, 120, 120]), saturation=3) == "indoor"


def test_frame_labels_around_the_onset(monkeypatch):
    val = load("umn_validation", monkeypatch)
    clip = {"panic_frame": 100, "label_end_frame": 150}
    assert val.frame_label(99, clip) == "normal"  # the burst 95-99 ends before the onset
    assert val.frame_label(100, clip) == "transition"  # 96-100 spans it
    assert val.frame_label(103, clip) == "transition"
    assert val.frame_label(104, clip) == "panic"  # 100-104 starts at the onset
    assert val.frame_label(150, clip) == "panic"
    assert val.frame_label(151, clip) == "after caption"
