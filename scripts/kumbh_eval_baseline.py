"""Counting accuracy, step 2: YOLO, CSRNet and the hybrid on the Kumbh evaluation frames, as deployed.

    venv\\Scripts\\python.exe scripts\\kumbh_eval_baseline.py

Ground truth: ml.kumbh_eval.load_ground_truth: per frame, the mean of every annotator's complete
row in data/eval/annotations/kumbh_eval/counts_<annotator>.csv (heads counted per cell of a 3x3
grid over the frame's counting region), else point files; frames marked unreadable are left out.
The report also shows how far the annotators agree on each frame.

Models exactly as the live system runs them (settings from .env): CSRNet with CSRNET_WEIGHTS at
INFER_MAX_WIDTH, YOLO person detection with YOLO_WEIGHTS / YOLO_IMGSZ / YOLO_CONF, and the hybrid
rule app.pipeline.choose_count (HYBRID_RULE, HYBRID_THRESHOLD), decided, as live, on the
whole-frame counts. Every method is then scored inside the frame's counting region: CSRNet's
density summed over the map cells whose centres lie in it, YOLO's people counted by their
estimated head position (ml.kumbh_eval.boxes_in_region), the hybrid by whichever it chose.
Count error only: counts.csv holds no head positions, so localisation is not evaluated.

Writes results/kumbh_eval_baseline.md (+ _frames.csv and _summary.csv). Changes no model or rule.
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import DATA_DIR, get_settings  # noqa: E402
from app.csrnet_model import TRUSTED_KINDS, DensityEstimator  # noqa: E402
from app.detector import load_detector  # noqa: E402
from app.image_io import read_image  # noqa: E402
from app.pipeline import choose_count  # noqa: E402
from ml.kumbh_eval import agreement, boxes_in_region, density_in_region, load_ground_truth, score  # noqa: E402

FRAMES = DATA_DIR / "eval" / "frames" / "kumbh_eval"
ANNOTATIONS = DATA_DIR / "eval" / "annotations" / "kumbh_eval"
OUT = ROOT / "results" / "kumbh_eval_baseline"
METHODS = ("CSRNet", "YOLO", "hybrid")
CYCLE_S, CAMERAS = 2.0, 3


def table(rows: list[dict]) -> list[str]:
    columns = list(rows[0])
    return ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns),
            *("| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows)]


def main() -> None:
    s = get_settings()
    estimator = DensityEstimator(s.csrnet_weights, s.device, s.infer_max_width, allow_pickle=s.csrnet_allow_pickle)
    if estimator.info["kind"] not in TRUSTED_KINDS:
        raise SystemExit(f"refusing {estimator.info['kind']} weights ({s.csrnet_weights})")
    detector, why = load_detector(s.yolo_weights, s.device, s.yolo_imgsz, s.yolo_conf)
    if detector is None:
        raise SystemExit(why)
    truth, notes = load_ground_truth(FRAMES, ANNOTATIONS)
    if not truth:
        raise SystemExit("no complete ground truth: count frames with tools.count_eval first")
    print("CSRNet:", estimator.info["label"])
    print("YOLO:", detector.info)
    for note in notes:
        print("note:", note)

    warm = read_image(FRAMES / truth[0].frame)
    for _ in range(3):  # warm-up: CUDA kernels and the ultralytics predictor
        estimator.predict(warm)
        detector.detect(warm)
    predictions = {m: {} for m in METHODS}
    rows, ms = [], {"CSRNet": [], "YOLO": []}
    for gt in truth:
        image = read_image(FRAMES / gt.frame)
        h, w = image.shape[:2]
        started = time.perf_counter()
        density = estimator.predict(image)
        ms["CSRNet"].append((time.perf_counter() - started) * 1e3)
        detections = detector.detect(image)
        ms["YOLO"].append(detections.ms)
        method = choose_count(density.count, detections.count, s.hybrid_threshold, s.hybrid_rule)
        csrnet = density_in_region(density.density, (w, h), gt.region)
        yolo = float(boxes_in_region(detections.boxes, gt.region))
        predictions["CSRNet"][gt.frame], predictions["YOLO"][gt.frame] = csrnet, yolo
        predictions["hybrid"][gt.frame] = yolo if method == "detection" else csrnet
        rows.append({"frame": gt.frame, "whole_frame_csrnet": round(density.count, 1),
                     "whole_frame_yolo": detections.count, "hybrid_uses": "YOLO" if method == "detection" else "CSRNet"})
        print(f"{gt.frame}: true {gt.count:g} | CSRNet {csrnet:.1f} | YOLO {yolo:.0f} | hybrid -> "
              f"{rows[-1]['hybrid_uses']} (whole frame: CSRNet {density.count:.1f}, YOLO {detections.count})")

    scored = {m: score(truth, predictions[m]) for m in METHODS}
    frame_rows = []
    for i, gt in enumerate(truth):
        x0, y0, x1, y1 = gt.region
        row = {"frame": gt.frame, "shot_type": gt.shot_type, "region (x0,y0,x1,y1)": f"{x0:g},{y0:g},{x1:g},{y1:g}",
               "true (annotators' mean)": f"{gt.count:g}"}
        for m in METHODS:
            f = scored[m][0][i]
            row[m] = f["predicted"]
            row[f"{m} err %"] = f["pct_error"]
        row["hybrid uses"] = rows[i]["hybrid_uses"]
        frame_rows.append(row)
    summary_rows = []
    for m in METHODS:
        for srow in scored[m][1]:
            if srow["shot_type"] == "all" and len(scored[m][1]) == 2:
                continue  # one shot type only: "all" would repeat it
            summary_rows.append({"method": m, "shot_type": srow["shot_type"], "frames": srow["frames"],
                                 "true_heads": f"{srow['true_heads']:g}", "MAE": srow["mae"],
                                 "relative error %": srow["relative_error_pct"],
                                 "ms per frame": round(statistics.mean(ms["CSRNet"]) + statistics.mean(ms["YOLO"]), 1)
                                 if m == "hybrid" else round(statistics.mean(ms[m]), 1)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix, data in (("_frames", frame_rows), ("_summary", summary_rows)):
        with open(OUT.with_name(OUT.name + suffix + ".csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    shots = sorted({gt.shot_type for gt in truth})
    hybrid_ms = statistics.mean(ms["CSRNet"]) + statistics.mean(ms["YOLO"])
    device = str(estimator.device)
    if estimator.device.type == "cuda":
        import torch

        device += f" ({torch.cuda.get_device_name(estimator.device)})"
    agree = agreement(truth)
    pairs = [float(r["difference %"]) for r in agree if r["difference %"] != "-"]
    agreement_note = (f"Frames counted by two or more people: {len(pairs)} of {len(agree)}; mean difference "
                      f"{sum(pairs) / len(pairs):.1f}%." if pairs else
                      "Every frame so far has a single annotator, so agreement cannot be measured yet.")
    left_out = [n for n in notes if "left out" in n]
    other_notes = [n for n in notes if "left out" not in n]
    n = len(truth)
    under = {m: sum(predictions[m][gt.frame] < gt.count for gt in truth) for m in METHODS}
    yolo_zero = sum(predictions["YOLO"][gt.frame] == 0 for gt in truth)
    uses_csrnet = sum(r["hybrid_uses"] == "CSRNet" for r in rows)
    observations = [
        f"- CSRNet predicted fewer heads than counted on {under['CSRNet']} of {n} frames, YOLO on {under['YOLO']}, "
        f"the hybrid on {under['hybrid']}.",
        f"- YOLO found nobody inside the region on {yolo_zero} of {n} frames.",
        f"- The hybrid used CSRNet's count on {uses_csrnet} of {n} frames and YOLO's on {n - uses_csrnet}.",
    ]
    lines = [
        "# Counting accuracy on Kumbh footage: baseline (step 2)", "",
        f"{len(truth)} frame(s) with complete human ground truth, shot types: {', '.join(shots)}. "
        "Each model is scored inside the frame's counting region; see the notes before reading the numbers.", "",
        "## Error per frame", "", *table(frame_rows), "",
        "## Error per shot type", "", *table(summary_rows), "",
        "## Inter-annotator agreement", "",
        *(table(agree) if agree else ["No frame counted with `tools.count_eval`."]), "",
        "Each person's total for the frame's region, the mean used as the true count, and the difference: "
        "(highest - lowest) / mean, which for two people is their % difference. " + agreement_note, "",
        "## Notes", "",
        "- **Ground truth**: the mean of the annotators' totals; each person counted the heads per cell of a 3x3 grid "
        "over the frame's counting region (`tools.count_eval --annotator <name>`, "
        "`data/eval/annotations/kumbh_eval/counts_<name>.csv`), without seeing anyone else's counts; no model "
        "suggested any count.",
        "- **Count error only**: the ground truth has no head positions, so point-level localisation is not "
        "evaluated. err % = |predicted - true| / true per frame; relative error % = sum |error| / sum true heads.",
        f"- **CSRNet**: {estimator.info['label']} At INFER_MAX_WIDTH={s.infer_max_width} (these 640x360 frames "
        "run at 640x360: frames are only ever downscaled); density summed over the map cells whose centres lie "
        "in the region.",
        f"- **YOLO**: {detector.info['model']}, person class, imgsz {s.yolo_imgsz}, conf {s.yolo_conf} "
        f"({detector.info['source']}); a person counts in the region by the estimated head position (box centre, "
        "0.2 x box width below its top), since the ground truth counts heads.",
        f"- **Hybrid**: `choose_count` with HYBRID_RULE={s.hybrid_rule}, HYBRID_THRESHOLD={s.hybrid_threshold:g}, "
        "decided as live on the whole-frame counts (shown below), then scored on the chosen model's count in the "
        "region.",
        f"- **Time**: mean model time per frame after warm-up on {device}, one frame at a time; the hybrid runs both "
        f"models. {CAMERAS} cameras in the {CYCLE_S:g} s cycle need {CAMERAS} x {hybrid_ms:.0f} ms = "
        f"{CAMERAS * hybrid_ms / 1e3:.2f} s of the hybrid's model time (optical flow and the rest of the cycle "
        "not included).",
        "- **Left out**: " + ("; ".join(left_out) if left_out else "nothing") + ".",
        *(f"- {note}." for note in other_notes),
        "",
        "## Observations (computed from the table)", "", *observations, "",
        "Whole-frame counts behind each hybrid decision:", "", *table(rows), "",
    ]
    if len(truth) < 5 or len(shots) < 2:
        lines.insert(4, f"> **Too little ground truth to compare methods in general.** Only {len(truth)} frame(s), "
                        f"all {', '.join(shots)}, are counted completely; the other shot types have no ground truth "
                        "yet, so this table says nothing about them, and nothing about dense crowds (the aerial "
                        "regions are the sparse sand and water areas). Count more frames with `tools.count_eval --annotator <name>` and "
                        "run this again.\n")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT.with_suffix('.md')}")


if __name__ == "__main__":
    main()
