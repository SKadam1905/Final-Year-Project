"""Part A: optical-flow speed against accuracy at different flow widths.

    venv\\Scripts\\python.exe scripts\\flow_benchmark.py

Takes a 5-frame burst every 12 frames of the test clip (scripts/make_test_clip.py) and runs
the production turbulence code (ml.optical_flow.burst_turbulence) with the flow computed at
full resolution and at 640, 480, 320 and 240 px wide. Writes to results/:
    flow_speed_vs_accuracy.csv / .md   time per burst, speed-up, agreement with full resolution
    flow_turbulence_per_burst.csv      every burst's score at every width
    flow_speed_vs_accuracy.png         time per burst and score agreement
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from ml.optical_flow import FlowParams, burst_turbulence  # noqa: E402
from report_utils import save_figure, write_table  # noqa: E402

REQUIRED_R = 0.95
PREFERRED_WIDTH = 320  # kept as the default if it meets REQUIRED_R, else the smallest width that does


def read_clip(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}; run scripts\\make_test_clip.py first")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return frames, fps


def regime_at(t: float, regimes: list[tuple[str, float, float]]) -> str:
    return next((name for name, start, end in regimes if start <= t < end), "")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", type=Path, default=ROOT / "data" / "test_clips" / "synthetic_crowd_1280.mp4")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--every", type=int, default=12, help="frames between burst starts")
    args = p.parse_args(argv)

    frames, fps = read_clip(args.clip)
    full = frames[0].shape[1]
    widths = [full] + [w for w in (640, 480, 320, 240) if w < full]
    regimes_file = args.clip.with_suffix(".regimes.csv")
    regimes = []
    if regimes_file.exists():
        with open(regimes_file) as f:
            regimes = [(r["regime"], float(r["start_s"]), float(r["end_s"])) for r in csv.DictReader(f)]
    starts = list(range(0, len(frames) - 5, args.every))
    print(f"{args.clip.name}: {len(frames)} frames {full}x{frames[0].shape[0]}, {len(starts)} bursts, "
          f"widths {widths}, OpenCV threads {cv2.getNumThreads()}")

    for w in widths:  # warm-up (first-call allocations), not timed
        burst_turbulence(frames[:5], flow_params=FlowParams(max_width=w))

    scores = {w: [] for w in widths}
    times = {w: [] for w in widths}
    for s in starts:
        burst = frames[s : s + 5]
        for w in widths:
            t0 = time.perf_counter()
            scores[w].append(burst_turbulence(burst, flow_params=FlowParams(max_width=w)).score)
            times[w].append((time.perf_counter() - t0) * 1000)

    ref = np.asarray(scores[full])
    rows = []
    for w in widths:
        s, t = np.asarray(scores[w]), np.asarray(times[w])
        r = float(np.corrcoef(ref, s)[0, 1]) if w != full else 1.0
        rows.append({
            "flow_width_px": w if w != full else f"{w} (full)",
            "mean_ms_per_burst": round(float(t.mean()), 1),
            "p95_ms_per_burst": round(float(np.percentile(t, 95)), 1),
            "speedup_vs_full": round(float(np.mean(times[full]) / t.mean()), 2),
            "speedup_vs_640": round(float(np.mean(times[640]) / t.mean()), 2) if 640 in times else "",
            "pearson_r_vs_full": round(r, 4),
            "mean_abs_diff_vs_full": round(float(np.abs(s - ref).mean()), 4),
            "max_abs_diff_vs_full": round(float(np.abs(s - ref).max()), 4),
            "meets_r_0.95": "yes" if r >= REQUIRED_R else "no",
        })
    passing = [w for w in widths[1:] if np.corrcoef(ref, scores[w])[0, 1] >= REQUIRED_R]
    choice = PREFERRED_WIDTH if PREFERRED_WIDTH in passing else (min(passing) if passing else full)

    args.out.mkdir(parents=True, exist_ok=True)
    notes = [
        f"Clip: {args.clip.name} ({full}x{frames[0].shape[0]}, {len(starts)} bursts of 5 consecutive frames, "
        f"one every {args.every} frames). Synthetic crowd with eight motion regimes; see scripts/make_test_clip.py.",
        "Time per burst = ml.optical_flow.burst_turbulence (downscale, cut check, 4 Farneback pairs, "
        f"turbulence) on one CPU thread calling OpenCV ({cv2.getNumThreads()} OpenCV threads).",
        f"Agreement is with the full-resolution score over the same bursts. Widths meeting r >= {REQUIRED_R}: "
        f"{', '.join(map(str, passing)) or 'none'}. Rule: keep {PREFERRED_WIDTH} px if it meets the bar, otherwise "
        f"use the smallest width that does. Chosen FLOW_MAX_WIDTH: {choice} px.",
        "The clip's regimes span turbulence 0-1, which helps the correlation; the mean absolute difference is the "
        "better measure of error. Re-check on real footage once it is available.",
    ]
    write_table(args.out / "flow_speed_vs_accuracy", rows, "Optical flow: speed against accuracy by flow width", notes)

    by_regime = []
    for name, _, _ in regimes:
        idx = [i for i, s in enumerate(starts) if regime_at(s / fps, regimes) == name]
        if idx:
            by_regime.append({"regime": name, "bursts": len(idx),
                              **{f"mean_turbulence_{w}": round(float(np.mean([scores[w][i] for i in idx])), 3)
                                 for w in widths}})
    if by_regime:
        write_table(args.out / "flow_turbulence_by_regime", by_regime, "Mean turbulence per motion regime and flow width",
                    ["Sanity check of the metric: still and orderly motion should score near 0, counter-flow and "
                     "chaotic motion near 1. Regimes are described in scripts/make_test_clip.py."])

    with open(args.out / "flow_turbulence_per_burst.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["burst_start_s", "regime", *(f"turbulence_{w}" for w in widths), *(f"ms_{w}" for w in widths)])
        for i, s in enumerate(starts):
            writer.writerow([round(s / fps, 2), regime_at(s / fps, regimes),
                             *(round(scores[w][i], 4) for w in widths), *(round(times[w][i], 1) for w in widths)])

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [f"{w}\n(full)" if w == full else str(w) for w in widths]
    means = [np.mean(times[w]) for w in widths]
    bars = ax1.bar(labels, means, color=["#475569" if w != choice else "#0f766e" for w in widths])
    for bar, w in zip(bars, widths):
        ax1.annotate(f"{np.mean(times[full]) / np.mean(times[w]):.1f}x", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha="center", va="bottom", fontsize=9)
    ax1.set_xlabel("Flow width (px)")
    ax1.set_ylabel("Time per 5-frame burst (ms)")
    ax1.set_title("(a) Optical-flow time per burst (speed-up vs full)")
    for w, marker in zip(widths[1:], "osD^"):
        r = np.corrcoef(ref, scores[w])[0, 1]
        ax2.scatter(ref, scores[w], s=14, marker=marker, alpha=0.7, label=f"{w} px (r = {r:.3f})")
    ax2.plot([0, 1], [0, 1], color="black", linewidth=1, label="y = x")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Turbulence at full resolution (0-1, unitless)")
    ax2.set_ylabel("Turbulence at reduced width (0-1, unitless)")
    ax2.set_title("(b) Agreement with full resolution")
    ax2.legend(fontsize=8, loc="upper left")
    save_figure(fig, args.out / "flow_speed_vs_accuracy.png")

    for row in rows:
        print(row)
    for row in by_regime:
        print(row)
    print(f"widths with r >= {REQUIRED_R}: {passing}; chosen FLOW_MAX_WIDTH: {choice} px")


if __name__ == "__main__":
    main()
