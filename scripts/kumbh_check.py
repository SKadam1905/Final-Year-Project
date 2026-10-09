"""Counts on real Kumbh footage with the active weights, for checking by eye (Phase 5B, steps 3-4).

    venv\\Scripts\\python.exe scripts\\kumbh_check.py

For every clip in data\\uploads\\videos and data\\india_dataset\\raw_videos, samples the video as
video analysis does (a 5-frame burst every 2 s of video, scene cuts skipped), runs CSRNet on the
middle frame and the optical flow across the burst, and writes to results\\kumbh_check\\:
    <clip>_grid.jpg      6 frames spread over the clip, each with its predicted count and where
                         CSRNet put the people (heatmap scaled per frame: it locates heads, it is not
                         a persons/m2 scale, which needs the camera's ground area)
    <clip>_samples.csv   every sample: time, count, whole-frame and density-weighted turbulence
and results\\kumbh_check.csv/.md: per clip the count range and how much it varies, and the two
turbulence variants side by side. Refuses to run with demo or untrained weights.
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import get_settings  # noqa: E402
from app.csrnet_model import DensityEstimator  # noqa: E402
from app.video_analysis import VIDEO_TYPES, bursts  # noqa: E402
from app.visualize import blend_rgba, colour_overlay, normalise_density  # noqa: E402
from ml.optical_flow import FlowParams, burst_has_cut, burst_turbulence  # noqa: E402
from report_utils import write_table  # noqa: E402

OUT = ROOT / "results" / "kumbh_check"
TILE_W = 640


def clips() -> list[tuple[str, Path]]:
    found = []
    uploads = ROOT / "data" / "uploads" / "videos"
    for meta in sorted(uploads.glob("*.json")):
        info = json.loads(meta.read_text(encoding="utf-8"))
        found.append((Path(info["filename"]).stem, Path(info["path"])))
    raw = ROOT / "data" / "india_dataset" / "raw_videos"
    found += [(p.stem, p) for p in sorted(raw.iterdir()) if p.suffix.lower() in VIDEO_TYPES] if raw.exists() else []
    return found


def tile(frame: np.ndarray, density: np.ndarray, t: float, count: float, flags: str) -> np.ndarray:
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (TILE_W, round(h * TILE_W / w)), interpolation=cv2.INTER_AREA)
    size = (frame.shape[1], frame.shape[0])
    frame = blend_rgba(frame, colour_overlay(normalise_density(density), size, max_alpha=0.5))
    cv2.rectangle(frame, (0, 0), (TILE_W, 34), (0, 0, 0), -1)
    text = f"t={int(t // 60)}:{int(t % 60):02d}  count {count:,.0f}{'  ' + flags if flags else ''}"
    cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main() -> None:
    s = get_settings()
    estimator = DensityEstimator(s.csrnet_weights, s.device, s.infer_max_width)
    if estimator.info["kind"] in ("demo", "untrained"):
        raise SystemExit(f"refusing to report counts with {estimator.info['kind']} weights ({s.csrnet_weights})")
    print("weights:", estimator.info["label"])
    flow_params = FlowParams(max_width=s.flow_max_width)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, path in clips():
        cap = cv2.VideoCapture(str(path))
        samples, kept = [], []
        for b in bursts(cap, s.burst_interval_s, s.burst_size):
            row = {"time_s": round(b.time_s, 2), "frame": b.frame_index}
            if burst_has_cut(b.frames, flow_params):
                samples.append({**row, "flag": "scene cut"})
                continue
            middle = b.frames[len(b.frames) // 2]
            result = estimator.predict(middle)
            frame_t = burst_turbulence(b.frames, flow_params=flow_params)
            weighted_t = burst_turbulence(b.frames, flow_params=flow_params, density=result.density)
            samples.append({**row, "count": round(result.count, 1), "turbulence_frame": round(frame_t.score, 4),
                            "turbulence_density": round(weighted_t.score, 4),
                            "camera_motion_px": round(frame_t.camera_motion, 2),
                            "flag": "moving camera" if frame_t.moving_camera else ""})
            kept.append((b.time_s, result.count, middle, result.density, samples[-1]["flag"]))
        cap.release()
        if not kept:
            print(f"{name}: no analysable samples")
            continue
        with open(OUT / f"{name}_samples.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["time_s", "frame", "count", "turbulence_frame", "turbulence_density",
                                                   "camera_motion_px", "flag"])
            writer.writeheader()
            writer.writerows(samples)
        picks = [kept[round(i * (len(kept) - 1) / 5)] for i in range(6)]
        tiles = [tile(frame, density, t, count, flags) for t, count, frame, density, flags in picks]
        grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
        cv2.imwrite(str(OUT / f"{name}_grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])

        counts = [k[1] for k in kept]
        analysed = [x for x in samples if "count" in x]
        tf = [x["turbulence_frame"] for x in analysed]
        td = [x["turbulence_density"] for x in analysed]
        r = float(np.corrcoef(tf, td)[0, 1]) if len(tf) > 2 and np.std(tf) > 0 and np.std(td) > 0 else float("nan")
        summary.append({
            "clip": name, "samples": len(samples), "analysed": len(analysed),
            "scene_cut": sum(x["flag"] == "scene cut" for x in samples),
            "moving_camera": sum(x["flag"] == "moving camera" for x in samples),
            "count_min": round(min(counts)), "count_median": round(st.median(counts)), "count_max": round(max(counts)),
            "distinct_counts": len({round(c) for c in counts}),
            "count_cv_pct": round(100 * st.pstdev(counts) / st.mean(counts), 1) if st.mean(counts) else "",
            "turbulence_frame_mean": round(st.mean(tf), 3), "turbulence_density_mean": round(st.mean(td), 3),
            "turbulence_mean_abs_diff": round(st.mean(abs(a - b) for a, b in zip(tf, td)), 3),
            "turbulence_pearson_r": round(r, 3),
        })
        print(summary[-1])
    write_table(ROOT / "results" / "kumbh_check", summary, "Counts on real Kumbh footage (active weights)", [
        f"Weights: {estimator.info['label']}",
        "No ground truth for these clips: the counts are for checking by eye with the grids in results/kumbh_check/; "
        "they are not an accuracy measurement.",
        "Sampling as in video analysis: one 5-frame burst every 2 s of video; bursts spanning a scene cut skipped. "
        "count_cv_pct = standard deviation / mean of the counts.",
        "turbulence_frame = every moving pixel counts (the default); turbulence_density = flow weighted by the "
        "CSRNet density map (TURBULENCE_WEIGHTING=density). Camera motion is removed first in both.",
    ])


if __name__ == "__main__":
    main()
