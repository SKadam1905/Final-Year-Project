"""Part D: how well does the turbulence score separate normal from panic crowd motion?

    venv\\Scripts\\python.exe scripts\\turbulence_roc.py --video data\\umn\\Crowd-Activity-All.avi --labels data\\umn\\labels.csv

labels.csv (template: scripts\\umn_labels_template.csv), one row per clip of the video:
    clip, scene, start_frame, end_frame, panic_frame
Frames are 0-based indices into the video file; panic_frame is the first frame where the crowd
starts to flee. Bursts of 5 consecutive frames are taken every --stride frames inside each clip and
scored with the production code (ml.optical_flow.burst_turbulence). A burst is "normal" if it ends
before panic_frame and "panic" if it starts at or after it; bursts spanning the transition are left
out. Writes to results/:
    turbulence_roc.csv / .md   per scene and overall: bursts, AUC, Youden-J threshold, and the
                               true/false positive rates at that threshold and at 0.3 and 0.6
    turbulence_roc.png         ROC curves per scene and overall
    raw/turbulence_scores.csv  every burst's score and label
The system's 0.3 / 0.6 turbulence bands are reported against, not changed.

--compare-density also scores every burst with the flow weighted by the CSRNet density map
(TURBULENCE_WEIGHTING=density, using the weights in CSRNET_WEIGHTS) and reports both AUCs, to
decide whether density weighting should become the default (only if its AUC is at least as high).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from ml.optical_flow import burst_turbulence  # noqa: E402
from report_utils import save_figure, write_table  # noqa: E402

SYSTEM_THRESHOLDS = (0.3, 0.6)


def read_labels(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = [{**r, **{k: int(r[k]) for k in ("start_frame", "end_frame", "panic_frame")}} for r in csv.DictReader(f)]
    for r in rows:
        if not r["start_frame"] < r["panic_frame"] <= r["end_frame"]:
            raise SystemExit(f"clip {r['clip']}: need start_frame < panic_frame <= end_frame")
    return rows


def score_bursts(video: Path, clips: list[dict], stride: int, estimator=None) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    bursts, recent, index = [], deque(maxlen=5), 0
    clip_of = {}
    for clip in clips:
        for f in range(clip["start_frame"], clip["end_frame"] + 1):
            clip_of[f] = clip
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        clip = clip_of.get(index)
        if clip is None:
            recent.clear()  # bursts never cross clip boundaries
        else:
            recent.append(frame)
            first = index - 4
            if len(recent) == 5 and first >= clip["start_frame"] and (first - clip["start_frame"]) % stride == 0:
                if index < clip["panic_frame"] or first >= clip["panic_frame"]:
                    burst = {"clip": clip["clip"], "scene": clip["scene"], "first_frame": first,
                             "panic": int(first >= clip["panic_frame"]),
                             "turbulence": burst_turbulence(list(recent)).score}
                    if estimator is not None:
                        density = estimator.predict(recent[2]).density
                        burst["turbulence_density"] = burst_turbulence(list(recent), density=density).score
                    bursts.append(burst)
        index += 1
    return bursts


def rates_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    predicted = scores >= threshold
    tpr = float(predicted[labels == 1].mean()) if (labels == 1).any() else float("nan")
    fpr = float(predicted[labels == 0].mean()) if (labels == 0).any() else float("nan")
    return tpr, fpr


def main(argv=None) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, default=ROOT / "data" / "umn" / "Crowd-Activity-All.avi")
    p.add_argument("--labels", type=Path, default=ROOT / "data" / "umn" / "labels.csv")
    p.add_argument("--stride", type=int, default=5, help="frames between burst starts")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--compare-density", action="store_true", help="also score density-weighted turbulence")
    args = p.parse_args(argv)

    estimator = None
    if args.compare_density:
        from app.config import get_settings
        from app.csrnet_model import DensityEstimator

        s = get_settings()
        estimator = DensityEstimator(s.csrnet_weights, s.device, s.infer_max_width)
        if estimator.info["kind"] in ("demo", "untrained"):
            raise SystemExit("density weighting needs real weights; CSRNET_WEIGHTS is a demo/untrained model")
    bursts = score_bursts(args.video, read_labels(args.labels), args.stride, estimator)
    raw = args.out / "raw" / "turbulence_scores.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    with open(raw, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bursts[0]))
        writer.writeheader()
        writer.writerows(bursts)

    import matplotlib.pyplot as plt

    groups = {f"scene: {s}": [b for b in bursts if b["scene"] == s] for s in sorted({b["scene"] for b in bursts})}
    groups["overall"] = bursts
    rows = []
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, group in groups.items():
        scores = np.array([b["turbulence"] for b in group])
        labels = np.array([b["panic"] for b in group])
        if len(set(labels)) < 2:
            rows.append({"group": name, "normal_bursts": int((labels == 0).sum()), "panic_bursts": int(labels.sum()),
                         "auc": "n/a (one class only)"})
            continue
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        best = int(np.argmax(tpr - fpr))
        row = {"group": name, "normal_bursts": int((labels == 0).sum()), "panic_bursts": int(labels.sum()),
               "auc": round(float(auc), 3), "youden_threshold": round(float(thresholds[best]), 3),
               "tpr_at_youden": round(float(tpr[best]), 3), "fpr_at_youden": round(float(fpr[best]), 3)}
        for t in SYSTEM_THRESHOLDS:
            t_tpr, t_fpr = rates_at(scores, labels, t)
            row[f"tpr_at_{t}"], row[f"fpr_at_{t}"] = round(t_tpr, 3), round(t_fpr, 3)
        rows.append(row)
        style = dict(linewidth=2.5, color="black") if name == "overall" else dict(linewidth=1.2)
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.2f})", **style)
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey", label="chance")
    ax.set_xlabel("False positive rate (normal bursts flagged)")
    ax.set_ylabel("True positive rate (panic bursts flagged)")
    ax.set_title("Turbulence score: normal vs panic (ROC)")
    ax.legend(fontsize=8, loc="lower right")
    save_figure(fig, args.out / "turbulence_roc.png")

    overall = next(r for r in rows if r["group"] == "overall")
    write_table(args.out / "turbulence_roc", rows, "Turbulence score as a panic detector (ROC)", [
        f"{len(bursts)} bursts of 5 consecutive frames, one every {args.stride} frames; bursts that span the "
        "normal-to-panic transition are excluded. Labels from " + args.labels.name + ".",
        "Youden's J picks the threshold that maximises TPR - FPR. The system currently labels turbulence low below "
        "0.3, medium below 0.6 and high from 0.6; those bands are not changed here.",
        f"Overall: AUC {overall.get('auc')}, best threshold {overall.get('youden_threshold')} "
        f"(vs 0.3 and 0.6 in the system).",
    ])
    for row in rows:
        print(row)
    if estimator is not None:
        labels = np.array([b["panic"] for b in bursts])
        both = [{"variant": name, "auc": round(float(roc_auc_score(labels, [b[key] for b in bursts])), 3)}
                for name, key in (("whole frame (default)", "turbulence"), ("density-weighted", "turbulence_density"))]
        write_table(args.out / "turbulence_weighting_auc", both, "Turbulence ROC AUC: whole frame vs density-weighted", [
            f"Same {len(bursts)} UMN bursts and labels; density from {estimator.info['label']}.",
            "Make TURBULENCE_WEIGHTING=density the default only if its AUC is at least as high.",
        ])
        print(both)


if __name__ == "__main__":
    main()
