"""Phase 6B: validate the turbulence score on the UMN Unusual Crowd Activity dataset.

    venv\\Scripts\\python.exe scripts\\umn_labels.py          # once: data\\umn\\labels.csv from the video's caption
    venv\\Scripts\\python.exe scripts\\umn_validation.py      # per-frame turbulence, ROC, delays, false alarms
    venv\\Scripts\\python.exe scripts\\umn_validation.py --reuse   # analysis only, from the saved per-frame scores

Per-frame score: for every frame t of a clip (from its 5th frame), the production function
ml.optical_flow.burst_turbulence on the 5-frame burst ending at t, with the live settings: Farneback as
in FlowParams, frames downscaled to FLOW_MAX_WIDTH (320 px; the UMN video is 320x240, so unchanged),
camera-motion compensation on, default TurbulenceParams. Bursts never cross a clip boundary.

Labels (published, from the video's caption; see scripts/umn_labels.py): a frame is "normal" when its
burst ends before panic_frame and "panic" when the burst starts at or after panic_frame and ends by
label_end_frame; bursts spanning the transition and frames after the caption are left out of the ROC.

Reported per scene and overall: ROC curve and AUC; the best threshold (Youden's J = TPR - FPR) with its
TPR / FPR; per clip the detection delay (seconds from panic_frame to the first frame at or after it with
score >= threshold); false alarms per minute of normal walking (switch-ons of score >= threshold within
normal frames). At the best overall threshold and at the system's turbulence bands, 0.3 (MEDIUM) and 0.6
(HIGH). Sensitivity: the same with the GUARD_S seconds just before each published onset left out, since
the caption appears after people start running in several clips. Reference: the burst's mean moving speed,
scored the same way (AUC only). Nothing in the live pipeline is changed.

Writes results/umn_validation.md, results/umn_validation_roc.png, results/umn_validation_timelines.png,
results/umn_validation_*.csv and results/raw/umn_turbulence_per_frame.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import get_settings  # noqa: E402
from ml.event_metrics import alarm_before, detection_delay, false_alarms  # noqa: E402
from ml.optical_flow import FlowParams, TurbulenceParams, burst_turbulence  # noqa: E402
from report_utils import save_figure  # noqa: E402

SYSTEM_BANDS = {"0.3 (MEDIUM band)": 0.3, "0.6 (HIGH band)": 0.6}
GUARD_S = 1.0
SCENES = ("lawn", "indoor", "plaza")
SCENE_COLOURS = {"lawn": "#2a78d6", "indoor": "#eb6834", "plaza": "#1baf7a"}  # validated, categorical slots 1-3
SCENE_DASHES = {"lawn": "-", "indoor": "--", "plaza": "-."}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"
PER_FRAME = ROOT / "results" / "raw" / "umn_turbulence_per_frame.csv"
# Not ground truth: looked at by eye on 2026-10-13 in data/umn/checks/*.jpg (frames 2 s, 1 s and 0.5 s before
# each caption onset). Listed for the user to confirm; the published caption is used regardless.
EYE_CHECK = {
    "people visibly running before the caption appears": [1, 2, 9, 10, 11],
    "running blur probably before the caption": [3, 6, 7],
    "no clear difference seen": [4, 5, 8],
}
FIELDS = ["clip", "scene", "frame", "clip_time_s", "label", "turbulence", "mean_speed", "moving_fraction",
          "camera_motion", "moving_camera"]


def read_labels(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("clip", "start_frame", "end_frame", "panic_frame", "label_end_frame"):
            r[k] = int(r[k])
    return rows


def frame_label(t: int, clip: dict, burst: int = 5) -> str:
    first = t - burst + 1
    if t < clip["panic_frame"]:
        return "normal"
    if first < clip["panic_frame"]:
        return "transition"
    return "panic" if t <= clip["label_end_frame"] else "after caption"


def score_frames(video: Path, clips: list[dict], flow: FlowParams, turb: TurbulenceParams) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    clip_of = {f: c for c in clips for f in range(c["start_frame"], c["end_frame"] + 1)}
    rows, recent, t, started = [], deque(maxlen=5), 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        clip = clip_of.get(t)
        if clip is None or t == clip["start_frame"]:
            recent.clear()  # bursts never cross clip boundaries
        if clip is not None:
            recent.append(frame)
            if len(recent) == 5:
                r = burst_turbulence(list(recent), None, flow, turb)
                rows.append({"clip": clip["clip"], "scene": clip["scene"], "frame": t,
                             "clip_time_s": round((t - clip["start_frame"]) / fps, 3), "label": frame_label(t, clip),
                             "turbulence": round(r.score, 5), "mean_speed": round(r.mean_speed, 4),
                             "moving_fraction": round(r.moving_fraction, 4), "camera_motion": round(r.camera_motion, 3),
                             "moving_camera": int(r.moving_camera)})
        t += 1
        if t % 500 == 0:
            print(f"  frame {t} ({time.perf_counter() - started:.0f} s)", flush=True)
    cap.release()
    return rows


def table(rows: list[dict]) -> list[str]:
    columns = list(rows[0])
    return ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns),
            *("| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows)]


def fmt_delay(d: float | None) -> str:
    return "missed" if d is None else f"{d:.2f}"


def main(argv=None) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, default=ROOT / "data" / "umn" / "Crowd-Activity-All.avi")
    p.add_argument("--labels", type=Path, default=ROOT / "data" / "umn" / "labels.csv")
    p.add_argument("--reuse", action="store_true", help="read the saved per-frame scores instead of recomputing")
    args = p.parse_args(argv)
    clips = read_labels(args.labels)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    s = get_settings()
    flow = FlowParams(max_width=s.flow_max_width, compensate_camera=True)
    turb = TurbulenceParams()

    if args.reuse and PER_FRAME.exists():
        with open(PER_FRAME, newline="", encoding="utf-8") as f:
            rows = [{**r, "clip": int(r["clip"]), "frame": int(r["frame"]), "turbulence": float(r["turbulence"]),
                     "mean_speed": float(r["mean_speed"]), "moving_camera": int(r["moving_camera"])}
                    for r in csv.DictReader(f)]
    else:
        print(f"scoring every frame of {len(clips)} clips (Farneback at {flow.max_width} px, camera compensation on)")
        started = time.perf_counter()
        rows = score_frames(args.video, clips, flow, turb)
        print(f"{len(rows)} frames scored in {time.perf_counter() - started:.0f} s")
        PER_FRAME.parent.mkdir(parents=True, exist_ok=True)
        with open(PER_FRAME, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    guard = round(GUARD_S * fps)
    onset_of = {c["clip"]: c["panic_frame"] for c in clips}

    def usable(r: dict, guarded: bool) -> bool:
        if r["label"] == "panic":
            return True
        return r["label"] == "normal" and not (guarded and r["frame"] >= onset_of[r["clip"]] - guard)

    groups = {**{s_: [r for r in rows if r["scene"] == s_] for s_ in SCENES}, "overall": rows}

    def roc(group: list[dict], key: str, guarded: bool):
        g = [r for r in group if usable(r, guarded)]
        y = np.array([r["label"] == "panic" for r in g], int)
        x = np.array([r[key] for r in g], float)
        fpr, tpr, thr = roc_curve(y, x)
        best = int(np.argmax(tpr - fpr))
        return {"auc": float(roc_auc_score(y, x)), "fpr": fpr, "tpr": tpr, "best": float(thr[best]),
                "best_tpr": float(tpr[best]), "best_fpr": float(fpr[best]), "y": y, "x": x,
                "normal": int((y == 0).sum()), "panic": int(y.sum())}

    def rates(res: dict, threshold: float) -> tuple[float, float]:
        on = res["x"] >= threshold
        return float(on[res["y"] == 1].mean()), float(on[res["y"] == 0].mean())

    results = {g: roc(groups[g], "turbulence", False) for g in groups}
    guarded = {g: roc(groups[g], "turbulence", True) for g in groups}
    speed = {g: roc(groups[g], "mean_speed", False) for g in groups}
    best = results["overall"]["best"]
    thresholds = {f"{best:.3f} (best, Youden)": best, **SYSTEM_BANDS}

    roc_rows = []
    for g in (*SCENES, "overall"):
        r = results[g]
        row = {"group": g, "normal frames": r["normal"], "panic frames": r["panic"], "AUC": f"{r['auc']:.3f}",
               "AUC, 1 s guard": f"{guarded[g]['auc']:.3f}", "best threshold": f"{r['best']:.3f}",
               "TPR / FPR at best": f"{r['best_tpr']:.2f} / {r['best_fpr']:.2f}"}
        for name, t in SYSTEM_BANDS.items():
            tpr, fpr = rates(r, t)
            row[f"TPR / FPR at {t}"] = f"{tpr:.2f} / {fpr:.2f}"
        row["mean speed AUC (reference)"] = f"{speed[g]['auc']:.3f}"
        roc_rows.append(row)

    def series(clip_no: int, labels: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        sel = [r for r in rows if r["clip"] == clip_no and (labels is None or r["label"] in labels)]
        return np.array([r["frame"] for r in sel]), np.array([r["turbulence"] for r in sel])

    delay_rows, delays = [], {name: {} for name in thresholds}
    for c in clips:
        frames, scores = series(c["clip"])
        row = {"clip": c["clip"], "scene": c["scene"], "panic from (s into clip)":
               f"{(c['panic_frame'] - c['start_frame']) / fps:.1f}"}
        for name, t in thresholds.items():
            d = detection_delay(frames, scores, c["panic_frame"], t, fps, last=c["label_end_frame"])
            delays[name][c["clip"]] = d
            early = alarm_before(frames, scores, c["panic_frame"], t, fps, GUARD_S)
            row[f"delay s @ {t:.3g}"] = fmt_delay(d) + (" (on before)" if early else "")
        delay_rows.append(row)

    fa_rows = []
    for g in (*SCENES, "overall"):
        group_clips = [c for c in clips if g in ("overall", c["scene"])]
        row = {"group": g}
        for guarded_flag in (False, True):
            minutes = 0.0
            counts = {name: 0 for name in thresholds}
            for c in group_clips:
                sel = [r for r in rows if r["clip"] == c["clip"] and r["label"] == "normal"
                       and not (guarded_flag and r["frame"] >= c["panic_frame"] - guard)]
                frames = np.array([r["frame"] for r in sel])
                scores = np.array([r["turbulence"] for r in sel])
                for name, t in thresholds.items():
                    fa = false_alarms(frames, scores, t, fps)
                    counts[name] += fa.count
                minutes += len(sel) / fps / 60
            suffix = ", 1 s guard" if guarded_flag else ""
            if not guarded_flag:
                row["normal minutes"] = f"{minutes:.2f}"
            for name, t in thresholds.items():
                row[f"FA/min @ {t:.3g}{suffix}"] = f"{counts[name] / minutes:.1f}"
        fa_rows.append(row)

    summary_rows = []
    for name, t in thresholds.items():
        found = [d for d in delays[name].values() if d is not None]
        tpr, fpr = rates(results["overall"], t)
        fa = next(r for r in fa_rows if r["group"] == "overall")
        summary_rows.append({"threshold": name, "panic frames flagged (TPR)": f"{tpr:.2f}",
                             "normal frames flagged (FPR)": f"{fpr:.2f}",
                             "clips detected": f"{len(found)} / {len(clips)}",
                             "median delay (s)": f"{np.median(found):.2f}" if found else "-",
                             "max delay (s)": f"{max(found):.2f}" if found else "-",
                             "false alarms / min of normal walking": fa[f"FA/min @ {t:.3g}"],
                             "with 1 s guard": fa[f"FA/min @ {t:.3g}, 1 s guard"]})

    out = ROOT / "results"
    for stem, data in (("umn_validation_roc", roc_rows), ("umn_validation_delays", delay_rows),
                       ("umn_validation_false_alarms", fa_rows), ("umn_validation_summary", summary_rows)):
        with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    plot_roc(results, thresholds, out / "umn_validation_roc.png")
    plot_timelines(rows, clips, thresholds, fps, out / "umn_validation_timelines.png")

    moving = sum(r["moving_camera"] for r in rows)
    overall = results["overall"]
    by_threshold = {r["threshold"]: r for r in summary_rows}
    best_row = next(r for name, r in by_threshold.items() if "best" in name)
    band_rows = [by_threshold[name] for name in SYSTEM_BANDS]
    fa_best = float(best_row["false alarms / min of normal walking"])
    per_scene_auc = ", ".join(f"{g} {results[g]['auc']:.2f}" for g in SCENES)
    per_scene_speed = ", ".join(f"{g} {speed[g]['auc']:.2f}" for g in SCENES)
    findings = [
        f"- **Separation**: overall AUC {overall['auc']:.3f} ("
        + ("a weak separator: well below the 0.8 usually asked of a detector" if overall["auc"] < 0.7 else
           "moderate" if overall["auc"] < 0.8 else "good")
        + f"); per scene {per_scene_auc}. Leaving out the second before "
        f"each published onset changes it to {guarded['overall']['auc']:.3f}.",
        f"- **The bands vs the data**: panic frames reach 0.3 in {band_rows[0]['panic frames flagged (TPR)']} of cases and "
        f"0.6 in {band_rows[1]['panic frames flagged (TPR)']}; 0.3 catches {band_rows[0]['clips detected']} panics and "
        f"0.6 {band_rows[1]['clips detected']}. On UMN both bands sit above almost all of the signal, normal or panic.",
        f"- **The best threshold is not usable as an alarm**: {best:.3f} flags {best_row['normal frames flagged (FPR)']} "
        f"of normal-walking frames, {fa_best:.0f} false alarms per minute on the raw per-frame score; it catches every "
        "panic only because the score already hovers around it during normal walking.",
        f"- **Speed separates better than turbulence here**: mean moving speed AUC {speed['overall']['auc']:.3f} overall "
        f"({per_scene_speed}). UMN's panic is people running away, each block "
        "moving one way: fast but locally orderly, which a direction-variance score is not built to see.",
    ]
    recommendation = (
        "Keep 0.3 / 0.6 as they are for now. Lowering the band to the UMN best threshold would mark a third of normal "
        "walking; raising it would never fire here. UMN does not test what the score is meant for (counter-flow, "
        "pushing and swirling in dense crowds), so it neither validates nor refutes the bands for Kumbh. It does show "
        "that turbulence alone does not detect a running crowd; a speed term (or a sudden-speed-rise alarm) looks more "
        "promising (speed AUCs above: high outdoors, low indoors) and should be evaluated before any change. Turbulence enters the live risk continuously (pressure = density x "
        f"(1 + {s.pressure_k:g} x turbulence)), so the bands only change the dashboard's LOW / MEDIUM / HIGH label."
        if overall["auc"] < 0.7 and fa_best > 10 else
        "Review the bands against the best threshold above before changing them; nothing is changed by this script.")
    lines = [
        "# Turbulence validation on UMN Unusual Crowd Activity (Phase 6B)", "",
        f"{len(clips)} clips (2 lawn, 6 indoor, 3 plaza), {len(rows)} frames scored at {fps:g} fps, {width}x{height}. "
        f"Overall AUC **{overall['auc']:.3f}** (normal walking vs panic, per frame); best threshold "
        f"{best:.3f} (TPR {overall['best_tpr']:.2f}, FPR {overall['best_fpr']:.2f}). The live pipeline is unchanged.", "",
        "## Comparison with the system's turbulence bands", "", *findings, "",
        f"**Recommendation (not applied)**: {recommendation}", "",
        "## At the best threshold and at the system's bands (overall)", "", *table(summary_rows), "",
        "TPR / FPR: share of panic / normal frames whose score reaches the threshold. A clip is detected when the "
        "score reaches the threshold between the published panic onset and the end of the caption; delay = seconds "
        "from the onset to that frame. False alarms: switch-ons of the alarm during normal walking, per minute.", "",
        "## ROC per scene", "", "![ROC](umn_validation_roc.png)", "", *table(roc_rows), "",
        f"AUC, 1 s guard: the same with the {GUARD_S:g} s before each published onset left out. Mean speed AUC: the "
        "burst's mean moving speed scored the same way, as a reference for what the turbulence score adds or misses.", "",
        "## Detection delay per clip", "", *table(delay_rows), "",
        f"\"(on before)\": the alarm was already on during the {GUARD_S:g} s before the published onset.", "",
        "## False alarms per minute of normal walking", "", *table(fa_rows), "",
        "## Per-frame turbulence per clip", "", "![timelines](umn_validation_timelines.png)", "",
        "## Ground truth", "",
        "The UMN video carries its ground truth as a red \"Abnormal Crowd Activity\" caption that the authors burnt "
        "into the frames from the moment they mark the crowd as panicking. `scripts/umn_labels.py` reads it frame by "
        "frame (no judgement); clip boundaries are the hard cuts after each caption; scenes from the clips' colours.",
        "", *table([{"clip": c["clip"], "scene": c["scene"], "frames": f"{c['start_frame']}-{c['end_frame']}",
                     "panic_frame (caption on)": c["panic_frame"], "caption until": c["label_end_frame"]}
                    for c in clips]), "",
        "**Eye check, for you to confirm** (not used as ground truth): the caption appears after people start running in "
        "several clips, which is why the 1 s guard results are shown. "
        + "; ".join(f"{what}: clips {', '.join(map(str, which))}" for what, which in EYE_CHECK.items())
        + ". Frames to look at: `data/umn/checks/onsets_clips_*.jpg` (git-ignored).", "",
        "## Method", "",
        f"- **Score**: `ml.optical_flow.burst_turbulence` (the production function) on the 5-frame burst ending at every "
        f"frame; FlowParams(max_width={flow.max_width}, camera compensation on), default TurbulenceParams (block "
        f"{turb.block} px at 640 px, min speed {turb.min_speed} px/frame). The video is {width} px wide, so the "
        f"{flow.max_width} px downscale leaves it unchanged. Bursts flagged as moving camera: {moving} of {len(rows)}.",
        "- **Labels**: normal = the burst ends before the onset; panic = it starts at or after the onset and ends while "
        "the caption shows; bursts spanning the onset and frames after the caption are left out of the ROC.",
        "- **Live sampling differs**: the live system scores one 5-frame burst every 2 s and smooths the risk over 30 s; "
        "this scores every frame, so delays and false-alarm counts are for the raw signal, before any smoothing.",
        "- **How the score is used live**: risk pressure = density x (1 + k x turbulence) with k = PRESSURE_K "
        f"({s.pressure_k:g}); turbulence is not thresholded there. 0.3 and 0.6 are the LOW / MEDIUM / HIGH bands of "
        "`turbulence_to_label` (backend/ml/optical_flow.py, frontend/src/risk.js) shown on the dashboard.",
    ]
    (out / "umn_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"overall AUC {overall['auc']:.3f}, best threshold {best:.3f} -> {out / 'umn_validation.md'}")


def plot_roc(results: dict, thresholds: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 1], [0, 1], linestyle=":", color=MUTED, linewidth=1, label="chance")
    for scene in SCENES:
        r = results[scene]
        ax.plot(r["fpr"], r["tpr"], color=SCENE_COLOURS[scene], linestyle=SCENE_DASHES[scene], linewidth=1.6,
                label=f"{scene} (AUC {r['auc']:.2f})")
    o = results["overall"]
    ax.plot(o["fpr"], o["tpr"], color=INK, linewidth=2.6, label=f"overall (AUC {o['auc']:.2f})")
    points: list[tuple[float, float, list[str]]] = []  # thresholds landing within 0.03 share one label
    for name, t in thresholds.items():
        on = o["x"] >= t
        fpr, tpr = float(on[o["y"] == 0].mean()), float(on[o["y"] == 1].mean())
        near = next((pt for pt in points if abs(pt[0] - fpr) < 0.03 and abs(pt[1] - tpr) < 0.03), None)
        if near:
            near[2].append(name.split(" ")[0])
        else:
            points.append((fpr, tpr, [name.split(" ")[0]]))
    for fpr, tpr, names in points:
        ax.plot(fpr, tpr, marker="o", markersize=8, color=INK, markerfacecolor="white", markeredgewidth=2, zorder=5)
        ax.annotate("t = " + " and ".join(names), (fpr, tpr), textcoords="offset points", xytext=(28, 26),
                    fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8),
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"), zorder=6)
    ax.set_xlabel("False positive rate (normal-walking frames flagged)", color=INK)
    ax.set_ylabel("True positive rate (panic frames flagged)", color=INK)
    ax.set_title("Turbulence score on UMN: ROC per scene", color=INK, loc="left")
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    save_figure(fig, path)


def plot_timelines(rows: list[dict], clips: list[dict], thresholds: dict, fps: float, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(12, 11), sharey=True)
    for ax, c in zip(axes.flat, clips):
        sel = [r for r in rows if r["clip"] == c["clip"]]
        t = np.array([r["frame"] for r in sel])
        x = (t - c["start_frame"]) / fps
        ax.axvspan((c["panic_frame"] - c["start_frame"]) / fps, (c["label_end_frame"] - c["start_frame"]) / fps,
                   color=GRID, alpha=0.7, linewidth=0)
        ax.plot(x, [r["turbulence"] for r in sel], color=SCENE_COLOURS["lawn"], linewidth=1.0)
        for name, value in thresholds.items():
            style = ":" if "best" in name else "--"
            ax.axhline(value, color=MUTED, linestyle=style, linewidth=1)
        ax.set_title(f"clip {c['clip']} ({c['scene']})", fontsize=9, color=INK, loc="left")
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.flat[len(clips):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("turbulence", color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("seconds into clip", color=INK)
    best = next(v for k, v in thresholds.items() if "best" in k)
    axes.flat[-1].text(0.02, 0.85, "grey band: published panic caption", fontsize=9, color=INK, transform=axes.flat[-1].transAxes)
    axes.flat[-1].text(0.02, 0.65, "dashed lines: system bands 0.3 / 0.6", fontsize=9, color=INK, transform=axes.flat[-1].transAxes)
    axes.flat[-1].text(0.02, 0.45, f"dotted line: best threshold {best:.3f}", fontsize=9, color=INK, transform=axes.flat[-1].transAxes)
    axes.flat[-1].text(0.02, 0.25, "blue: turbulence of the burst ending at each frame", fontsize=9, color=INK,
                       transform=axes.flat[-1].transAxes)
    fig.suptitle("Per-frame turbulence on the 11 UMN clips", x=0.01, ha="left", color=INK)
    save_figure(fig, path)


if __name__ == "__main__":
    main()
