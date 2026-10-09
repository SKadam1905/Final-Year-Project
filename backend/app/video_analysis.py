"""Analysis of an uploaded crowd video with the live cameras' pipeline.

Sampling copies the live cameras: every BURST_INTERVAL_S (2 s) of *video* time, a burst of
BURST_SIZE (5) consecutive frames; CSRNet on the middle frame, Farneback flow across the
burst, then a fresh RiskEngine with the live rolling window. Times come from each frame's own
timestamp in the file (CAP_PROP_POS_MSEC, falling back to frame index / fps), never from the
wall clock, so a video analyses the same way however fast the machine is.

A sample whose burst spans a scene cut is skipped (no reading, kept out of the rolling risk
average and the summary) and recorded as "scene cut". Camera motion is removed before
turbulence (ml.optical_flow); a sample where it was large is marked "moving camera".

Used by app.video_jobs (the background queue) and app.video_api (the endpoints).
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from app.alert_dispatcher import AlertDispatcher
from app.calibration import ZoneGeometry
from app.config import Settings, ZoneConfig
from app.exit_router import ExitRouter
from app.gate_actions import gate_actions
from app.pipeline import ZoneAnalyzer
from app.risk_engine import RiskLevel
from app.visualize import density_scale, draw_banner, draw_density_legend, overlay_density_m2

VIDEO_TYPES = (".mp4", ".avi", ".mov", ".mkv")
FRAME_WIDTH = 960  # stored sample frames (and their overlays) are at most this wide
ANNOTATED_WIDTH = 1280
CSV_COLUMNS = ["time_s", "frame", "count", "count_method", "density_count", "detection_count", "density_per_m2",
               "turbulence", "raw_pressure", "pressure", "risk_level", "readings_in_mean", "hotspot",
               "hotspot_count", "camera_motion_px", "flag", "inference_ms", "detection_ms", "flow_ms"]
SCENE_CUT, MOVING_CAMERA = "scene cut", "moving camera"  # values of the CSV "flag" column


class VideoRejected(ValueError):
    """An upload that is not a usable video; `status` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class Cancelled(Exception):
    """The job was deleted while it ran."""


def safe_filename(name: str, max_length: int = 80) -> str:
    """A file name safe on any OS: the base name only, letters/digits/._- kept, the rest '_'."""
    base = re.split(r"[\\/]", name or "")[-1].strip().strip(".")
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "video"
    ext = re.sub(r"[^A-Za-z0-9]+", "", ext).lower()
    stem = stem[: max_length - len(ext) - 1]
    return f"{stem}.{ext}" if ext else stem


def check_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in VIDEO_TYPES:
        raise VideoRejected(415, f"Unsupported file type '{ext or '(none)'}': upload an "
                                 f"{', '.join(t.lstrip('.') for t in VIDEO_TYPES[:-1])} or {VIDEO_TYPES[-1][1:]} video.")
    return ext


def probe_video(path: Path) -> dict:
    """Open the file with OpenCV and decode a frame; raises VideoRejected if it is not a video."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoRejected(422, "This file could not be opened as a video. It may be damaged, or use a "
                                     "codec OpenCV cannot read; try re-saving it as H.264 MP4.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoRejected(422, "This file opened but no frame could be decoded. It may be damaged, or "
                                     "use a codec OpenCV cannot read; try re-saving it as H.264 MP4.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 <= fps <= 240 else 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height, width = frame.shape[:2]
    finally:
        cap.release()
    if frames <= 0:
        frames = count_frames(path)
    return {"fps": round(fps, 3), "frames": frames, "duration_s": round(frames / fps, 2),
            "width": width, "height": height}


def count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


@dataclass(frozen=True)
class VideoBurst:
    frames: list[np.ndarray]
    start_s: float  # timestamp of the burst's first frame
    time_s: float  # timestamp of the middle frame: the one CSRNet analyses
    frame_index: int  # index of the middle frame in the file
    end_index: int  # index of the burst's last frame


def bursts(cap: cv2.VideoCapture, interval_s: float = 2.0, burst: int = 5) -> Iterator[VideoBurst]:
    """Bursts as a live camera gives them: every `interval_s` of video time, `burst` consecutive
    frames, starting at the first frame whose timestamp is at or after each multiple of
    `interval_s`. A burst cut short by the end of the file is dropped."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 240 else 25.0
    next_start, index, previous = 0.0, -1, -math.inf
    frames: list[np.ndarray] = []
    times: list[float] = []
    while cap.grab():
        index += 1
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not math.isfinite(t) or t <= previous or (index > 0 and t == 0):
            t = index / fps  # no usable timestamp in the container
        previous = t
        if not frames and t < next_start - 1e-6:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            frames, times = [], []
            continue
        frames.append(frame)
        times.append(t)
        if len(frames) == burst:
            middle = burst // 2
            yield VideoBurst(frames, times[0], times[middle], index - burst + 1 + middle, index)
            while next_start <= times[0] + 1e-6:
                next_start += interval_s
            frames, times = [], []


def _display_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= FRAME_WIDTH:
        return frame
    return cv2.resize(frame, (FRAME_WIDTH, round(h * FRAME_WIDTH / w)), interpolation=cv2.INTER_AREA)


def analyse_video(
    path: Path,
    zone: ZoneConfig,
    estimator,
    settings: Settings,
    out_dir: Path,
    dispatcher: AlertDispatcher,
    graph_node: str | None = None,
    progress: Callable[[float], None] = lambda percent: None,
    cancelled: Callable[[], bool] = lambda: False,
    wait_turn: Callable[[], None] = lambda: None,
    detector=None,
) -> dict:
    """Run the live pipeline over the whole file. Saves each sample's frame (JPEG, at most 960 px
    wide) and, unless it is a scene cut, its density map (float16 .npy) under `out_dir`, and
    returns the results: video info, per-sample readings, alerts, the summary and the exits at
    the peak. With a `detector` (app.detector.PersonDetector) counts are hybrid, as live, and each
    sample's person boxes are saved too (scaled to the stored frame)."""
    info = probe_video(path)
    frames_dir, density_dir = out_dir / "frames", out_dir / "density"
    frames_dir.mkdir(parents=True, exist_ok=True)
    density_dir.mkdir(parents=True, exist_ok=True)
    boxes_dir = out_dir / "detections"
    analyzer = ZoneAnalyzer(zone, estimator, settings, detector=detector)  # a fresh RiskEngine: same window as live
    router = ExitRouter()  # its own copy of the graph: other zones count as clear
    node = graph_node if graph_node in router.graph else None

    samples, alerts, starts, exits_by_sample = [], [], [], []
    cap = cv2.VideoCapture(str(path))
    try:
        for i, b in enumerate(bursts(cap, settings.burst_interval_s, settings.burst_size)):
            if cancelled():
                raise Cancelled()
            wait_turn()  # live cameras first
            a = analyzer.analyze(b.frames, b.time_s)
            middle = b.frames[len(b.frames) // 2]
            shown = _display_frame(middle)
            cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), shown, [cv2.IMWRITE_JPEG_QUALITY, 88])
            starts.append(b.start_s)
            if info["frames"]:
                progress(min(99.0, 100.0 * (b.end_index + 1) / info["frames"]))
            if a.scene_cut:  # two shots in one burst: no reading, the risk average is untouched
                samples.append({"index": i, "time_s": round(b.time_s, 3), "frame": b.frame_index, "scene_cut": True,
                                "moving_camera": False, "camera_motion": None, "count": None, "density": None,
                                "turbulence": None, "raw_pressure": None, "pressure": None, "risk_level": None,
                                "samples": None, "inference_ms": None, "flow_ms": round(a.flow_ms, 1),
                                "count_method": None, "density_count": None, "detection_count": None,
                                "detection_ms": None, "grid": None, "hotspot": None})
                exits_by_sample.append([])
                continue
            s, t = a.status, a.turbulence
            exits = []
            if node:
                router.update_pressure(node, s.smoothed_pressure)
                exits = router.get_best_exits(node)
            np.save(density_dir / f"{i:05d}.npy", a.density.density.astype(np.float16))
            if a.detections is not None:
                boxes_dir.mkdir(exist_ok=True)
                scale = shown.shape[1] / middle.shape[1]
                np.save(boxes_dir / f"{i:05d}.npy", (a.detections.boxes * scale).astype(np.float32))
            actions = gate_actions(s.level, router, node, exits, s.hotspot)
            event = dispatcher.handle(s, zone.name, exits[0]["path"] if exits else None, [a["text"] for a in actions])
            if event is not None:
                alerts.append({"time_s": round(b.time_s, 2), "sample": i, "risk_level": str(event.level),
                               "message": event.text, "sent": bool(event.deliveries),
                               "deliveries": [{"channel": d.channel, "delivered": d.ok, "error": d.error}
                                              for d in event.deliveries]})
            samples.append({
                "index": i, "time_s": round(b.time_s, 3), "frame": b.frame_index, "scene_cut": False,
                "moving_camera": t.moving_camera, "camera_motion": round(t.camera_motion, 2),
                "count": round(s.count, 2), "density": round(s.density, 4), "turbulence": round(s.turbulence, 4),
                "raw_pressure": round(s.pressure, 4), "pressure": round(s.smoothed_pressure, 4),
                "risk_level": str(s.level), "samples": s.samples,
                "inference_ms": round(a.inference_ms, 1), "flow_ms": round(a.flow_ms, 1),
                "count_method": s.count_method, "density_count": round(s.density_count, 2),
                "detection_count": s.detection_count, "detection_ms": round(a.detection_ms, 1),
                "grid": a.grid.to_dict(), "hotspot": s.hotspot, "gate_actions": actions,
            })
            exits_by_sample.append(exits)
    finally:
        cap.release()
    if not samples:
        raise VideoRejected(422, f"The video is too short: it needs at least {settings.burst_size} frames.")
    if all(sample["scene_cut"] for sample in samples):
        raise VideoRejected(422, "Every sample spans a scene cut, so nothing could be analysed.")

    summary = summarise(samples, starts, info["duration_s"])
    peak = summary["peak_pressure_sample"]
    return {
        "video": info,
        "zone": {"name": zone.name, "area_m2": zone.area_m2, "graph_node": node},
        "model": getattr(estimator, "info", {"kind": "untrained", "label": "Unknown model", "weights": None}),
        "detector": getattr(detector, "info", None),
        "settings": {"burst_interval_s": settings.burst_interval_s, "burst_size": settings.burst_size,
                     "rolling_window_s": settings.rolling_window_s, "rolling_maxlen": settings.rolling_maxlen,
                     "pressure_k": settings.pressure_k, "flow_max_width": settings.flow_max_width,
                     "infer_max_width": settings.infer_max_width,
                     "hybrid_threshold": settings.hybrid_threshold, "hybrid_rule": settings.hybrid_rule,
                     "grid": [settings.grid_rows, settings.grid_cols],
                     "thresholds": {"yellow": settings.risk_yellow, "orange": settings.risk_orange,
                                    "red": settings.risk_red},
                     "density_scale": density_scale(settings.risk_yellow, settings.risk_orange,
                                                    settings.risk_red).to_dict()},
        "summary": summary,
        "samples": samples,
        "alerts": alerts,
        "exits_at_peak": {"time_s": samples[peak]["time_s"], "pressure": samples[peak]["pressure"],
                          "graph_node": node, "exits": exits_by_sample[peak]},
    }


def summarise(samples: list[dict], starts: list[float], duration_s: float) -> dict:
    """Peaks, the highest level reached, and the seconds of video spent at each level (each
    sample's level holds from its burst's start to the next burst's start, or the video's end).
    Scene cuts are left out of all of it; their time is reported separately."""
    analysed = [i for i, s in enumerate(samples) if not s["scene_cut"]]

    def peak(key: str) -> int:
        return max(analysed, key=lambda i: (samples[i][key], -i))

    time_at = {str(level): 0.0 for level in RiskLevel}
    cut_time = 0.0
    for i, sample in enumerate(samples):
        span = (starts[i + 1] if i + 1 < len(samples) else max(duration_s, starts[i])) - starts[i]
        if sample["scene_cut"]:
            cut_time += span
        else:
            time_at[sample["risk_level"]] += span
    count_i, density_i, pressure_i = peak("count"), peak("density"), peak("pressure")
    highest = max((RiskLevel(samples[i]["risk_level"]) for i in analysed), key=lambda level: level.severity)
    return {
        "duration_s": duration_s, "samples": len(analysed),
        "scene_cut_samples": len(samples) - len(analysed),
        "moving_camera_samples": sum(1 for s in samples if s["moving_camera"]),
        "detection_samples": sum(1 for s in samples if s.get("count_method") == "detection"),
        "time_scene_cut_s": round(cut_time, 2),
        "peak_count": samples[count_i]["count"], "peak_count_time_s": samples[count_i]["time_s"],
        "peak_density": samples[density_i]["density"], "peak_density_time_s": samples[density_i]["time_s"],
        "peak_pressure": samples[pressure_i]["pressure"], "peak_pressure_time_s": samples[pressure_i]["time_s"],
        "peak_pressure_sample": pressure_i,
        "peak_raw_pressure": max(samples[i]["raw_pressure"] for i in analysed),
        "highest_level": str(highest),
        "time_at_level_s": {level: round(seconds, 2) for level, seconds in time_at.items()},
    }


def write_csv(results: dict, path: Path) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for s in results["samples"]:  # a scene cut has only its time, frame and flag
            flag = SCENE_CUT if s["scene_cut"] else MOVING_CAMERA if s["moving_camera"] else ""
            hotspot = s.get("hotspot") or {}
            writer.writerow([s["time_s"], s["frame"], s["count"], s.get("count_method"), s.get("density_count"),
                             s.get("detection_count"), s["density"], s["turbulence"], s["raw_pressure"],
                             s["pressure"], s["risk_level"], s["samples"], hotspot.get("label"), hotspot.get("count"),
                             s["camera_motion"], flag, s["inference_ms"], s.get("detection_ms"), s["flow_ms"]])
    return path


def _video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """H.264 (plays in browsers) when this OpenCV build can write it, else MPEG-4 Part 2."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("OpenCV cannot write MP4 video on this machine")


def render_annotated(
    video: Path,
    results: dict,
    density_dir: Path,
    out_path: Path,
    progress: Callable[[float], None] = lambda percent: None,
    cancelled: Callable[[], bool] = lambda: False,
    wait_turn: Callable[[], None] = lambda: None,
) -> Path:
    """The video with the latest sample's persons/m² heatmap (the fixed scale, with its legend),
    count and risk level drawn on every frame (at most 1280 px wide). Slow: every frame is
    decoded, drawn and re-encoded."""
    info, samples = results["video"], results["samples"]
    scale = density_scale(**results["settings"]["thresholds"])
    geometry = ZoneGeometry(area_m2=results["zone"]["area_m2"])
    cap = cv2.VideoCapture(str(video))
    w, h = info["width"], info["height"]
    if w > ANNOTATED_WIDTH:
        w, h = ANNOTATED_WIDTH, round(h * ANNOTATED_WIDTH / info["width"]) // 2 * 2
    tmp = out_path.with_suffix(".part.mp4")
    writer = _video_writer(tmp, info["fps"], (w, h))
    current, per_m2, index, n = -1, None, 0, max(info["frames"], 1)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % 25 == 0:
                if cancelled():
                    raise Cancelled()
                wait_turn()
            while current + 1 < len(samples) and samples[current + 1]["frame"] <= index:
                current += 1
                if not samples[current].get("scene_cut"):
                    density = np.load(density_dir / f"{current:05d}.npy").astype(np.float32)
                    per_m2 = geometry.cell_density(density, (w, h))
            if (frame.shape[1], frame.shape[0]) != (w, h):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            if current >= 0:
                s = samples[current]
                if s.get("scene_cut"):
                    draw_banner(frame, ["Scene cut: sample skipped", f"t={s['time_s']:.0f}s"])
                else:
                    frame = overlay_density_m2(frame, per_m2, scale)
                    camera = "  moving camera" if s.get("moving_camera") else ""
                    draw_banner(frame, [f"{s['risk_level']}  pressure {s['pressure']:.2f}{camera}",
                                        f"~{s['count']:.0f} people  {s['density']:.2f}/m2  turbulence "
                                        f"{s['turbulence']:.2f}  t={s['time_s']:.0f}s"], RiskLevel(s["risk_level"]))
                    draw_density_legend(frame, scale)
            writer.write(frame)
            index += 1
            if index % 25 == 0:
                progress(min(99.0, 100.0 * index / n))
    finally:
        cap.release()
        writer.release()
    tmp.replace(out_path)
    return out_path
