"""Counting accuracy, step 3: tiled / upscaled inference vs the pretrained models as deployed.

    venv\\Scripts\\python.exe scripts\\kumbh_eval_tiling.py

Same ground truth and scoring as scripts/kumbh_eval_baseline.py (ml.kumbh_eval: the mean of all
annotators' complete counts per frame, inside each frame's counting region, unreadable frames left
out; count error only). Variants (app.tiled_inference; the live pipeline is not changed):

  CSRNet 1x / 2x / 3x     CSRNet on the frame as is, or upscaled 2x / 3x (bicubic)
  YOLO                    whole frame, as deployed
  YOLO sliced             SAHI-style: 320 px tiles, 20% overlap, each upscaled to 640, plus the whole
                          frame; boxes merged across tiles (intersection over smaller box >= 0.5)
  hybrid ...              the live 'agree' rule (HYBRID_RULE, HYBRID_THRESHOLD) on the variant's
                          whole-frame counts, scored on the chosen model's count in the region

Time: model time per frame on this GPU, measured separately for each model variant over all frames
(as a deployed system runs one variant; interleaving 1x/2x/3x inputs slows every one of them): warm-up,
then the median of 3 runs per frame, averaged over the frames. On this laptop the result moved by up
to 2x between runs of the script (power and clock states), so this is repeated in TIMING_PASSES
separate passes and reported as the median pass with the range; a variant fits the budget only if
even its slowest pass does ("borderline": the median pass fits, the slowest does not). A hybrid's
time is the sum of its two models' in the same pass.
Budget: 4 cameras in the 2 s cycle = 500 ms per frame (optical flow and the rest not included).
Writes results/kumbh_eval_tiling.md (+ _frames.csv, _summary.csv).
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import DATA_DIR, get_settings  # noqa: E402
from app.csrnet_model import TRUSTED_KINDS, DensityEstimator  # noqa: E402
from app.detector import load_detector  # noqa: E402
from app.image_io import read_image  # noqa: E402
from app.pipeline import choose_count  # noqa: E402
from app.tiled_inference import sliced_detect, tiles, upscaled_density, yolo_batch  # noqa: E402
from ml.kumbh_eval import agreement, boxes_in_region, density_in_region, load_ground_truth, score  # noqa: E402

FRAMES = DATA_DIR / "eval" / "frames" / "kumbh_eval"
ANNOTATIONS = DATA_DIR / "eval" / "annotations" / "kumbh_eval"
OUT = ROOT / "results" / "kumbh_eval_tiling"
CAMERAS, CYCLE_MS = 4, 2000.0
TILE, OVERLAP, UPSCALE_TO = 320, 0.2, 640
FACTORS = (1, 2, 3)
REPEATS = 3
TIMING_PASSES = 3
SINGLE = ["CSRNet 1x", "CSRNet 2x", "CSRNet 3x", "YOLO", "YOLO sliced"]
HYBRIDS = {"hybrid (pretrained)": ("CSRNet 1x", "YOLO"),
           "hybrid 1x + sliced": ("CSRNet 1x", "YOLO sliced"),
           "hybrid 2x + sliced": ("CSRNet 2x", "YOLO sliced"),
           "hybrid 3x + sliced": ("CSRNet 3x", "YOLO sliced")}
VARIANTS = SINGLE + list(HYBRIDS)
PRETRAINED = {"CSRNet 1x", "YOLO", "hybrid (pretrained)"}


def table(rows: list[dict]) -> list[str]:
    columns = list(rows[0])
    return ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns),
            *("| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows)]


def timed(fn, repeats: int = REPEATS):
    """(last result, median wall time in ms) of `repeats` calls."""
    import time

    times, result = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - started) * 1e3)
    return result, statistics.median(times)


def main() -> None:
    s = get_settings()
    # max_width=None: frames are fed at the size given (the live 1280 limit would shrink the 3x frames back)
    estimator = DensityEstimator(s.csrnet_weights, s.device, None, allow_pickle=s.csrnet_allow_pickle)
    if estimator.info["kind"] not in TRUSTED_KINDS:
        raise SystemExit(f"refusing {estimator.info['kind']} weights ({s.csrnet_weights})")
    detector, why = load_detector(s.yolo_weights, s.device, s.yolo_imgsz, s.yolo_conf)
    if detector is None:
        raise SystemExit(why)
    batch = yolo_batch(detector)
    truth, notes = load_ground_truth(FRAMES, ANNOTATIONS)
    if not truth:
        raise SystemExit("no complete ground truth: count frames with tools.count_eval first")

    runners = {f"CSRNet {f}x": (lambda image, f=f: upscaled_density(estimator, image, f)) for f in FACTORS}
    runners["YOLO"] = detector.detect

    def sliced(image):
        whole = detector.detect(image)
        return sliced_detect(batch, image, (whole.boxes, whole.scores), TILE, OVERLAP, UPSCALE_TO)

    runners["YOLO sliced"] = sliced
    images = {gt.frame: read_image(FRAMES / gt.frame) for gt in truth}

    # 1. Counts: every model variant once per frame.
    predictions = {v: {} for v in VARIANTS}
    decisions = []
    for gt in truth:
        image = images[gt.frame]
        h, w = image.shape[:2]
        whole_count = {}
        for v in SINGLE:
            result = runners[v](image)
            whole_count[v] = result.count
            predictions[v][gt.frame] = (density_in_region(result.density, (w, h), gt.region) if v.startswith("CSRNet")
                                        else float(boxes_in_region(result.boxes, gt.region)))
        row = {"frame": gt.frame}
        for v, (dens, det) in HYBRIDS.items():
            method = choose_count(whole_count[dens], int(whole_count[det]), s.hybrid_threshold, s.hybrid_rule)
            predictions[v][gt.frame] = predictions[det if method == "detection" else dens][gt.frame]
            row[v] = det if method == "detection" else dens
        decisions.append(row)
        print(gt.frame, f"true {gt.count:g}", {v: round(predictions[v][gt.frame], 1) for v in VARIANTS})

    # 2. Time: each model variant on its own over all frames, after a warm-up; several separate passes.
    passes = {v: [] for v in VARIANTS}  # mean ms per frame, one value per pass
    for n in range(TIMING_PASSES):
        for v in SINGLE:
            for _ in range(3):
                runners[v](images[truth[0].frame])
            passes[v].append(statistics.mean(timed(lambda: runners[v](images[gt.frame]))[1] for gt in truth))
        for v, (dens, det) in HYBRIDS.items():
            passes[v].append(passes[dens][-1] + passes[det][-1])
        print(f"timing pass {n + 1}:", {v: round(passes[v][-1]) for v in SINGLE})

    scored = {v: score(truth, predictions[v]) for v in VARIANTS}
    shots = sorted({gt.shot_type for gt in truth}) + ["all"]
    mean_ms = {v: statistics.median(passes[v]) for v in VARIANTS}  # the median pass
    budget = CYCLE_MS / CAMERAS

    def fits(v: str) -> str:
        return "yes" if max(passes[v]) <= budget else "borderline" if mean_ms[v] <= budget else "no"

    summary = []
    for v in VARIANTS:
        for srow in scored[v][1]:
            group = [gt for gt in truth if srow["shot_type"] in ("all", gt.shot_type)]
            signed = sum(predictions[v][gt.frame] - gt.count for gt in group)
            summary.append({"variant": v, "shot_type": srow["shot_type"], "frames": srow["frames"],
                            "true_heads": f"{srow['true_heads']:g}", "MAE": srow["mae"],
                            "relative_error_pct": srow["relative_error_pct"],
                            "bias_pct": round(100 * signed / srow["true_heads"], 1) if srow["true_heads"] else "",
                            "ms_per_frame": round(mean_ms[v], 1)})
    by = {(r["variant"], r["shot_type"]): r for r in summary}
    comparison = [{"shot type": shot, "frames": by[(VARIANTS[0], shot)]["frames"],
                   "true heads": by[(VARIANTS[0], shot)]["true_heads"],
                   **{v: f"{by[(v, shot)]['MAE']} ({by[(v, shot)]['relative_error_pct']}%)" for v in VARIANTS}}
                  for shot in shots]
    comparison.append({"shot type": "**ms per frame** (median pass)", "frames": "", "true heads": "",
                       **{v: f"{mean_ms[v]:.0f}" for v in VARIANTS}})
    comparison.append({"shot type": f"**{CAMERAS} cameras in {CYCLE_MS / 1e3:g} s**", "frames": "", "true heads": "",
                       **{v: fits(v) for v in VARIANTS}})
    bias = [{"shot type": shot, **{v: f"{by[(v, shot)]['bias_pct']:+}%" if by[(v, shot)]["bias_pct"] != "" else "-"
                                   for v in VARIANTS}} for shot in shots]
    timing = [{"variant": v, "ms per frame (median pass)": round(mean_ms[v], 1),
               "range over passes": f"{min(passes[v]):.0f}-{max(passes[v]):.0f}",
               f"{CAMERAS} cameras (s), slowest pass": round(CAMERAS * max(passes[v]) / 1e3, 2),
               f"within the {CYCLE_MS / 1e3:g} s cycle": fits(v),
               "kind": "pretrained, as deployed" if v in PRETRAINED else "tiled / upscaled"} for v in VARIANTS]
    frame_rows = [{"frame": gt.frame, "shot_type": gt.shot_type, "true (annotators' mean)": f"{gt.count:g}",
                   **{v: round(predictions[v][gt.frame], 1) for v in VARIANTS}} for gt in truth]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix, data in (("_frames", frame_rows), ("_summary", summary)):
        with open(OUT.with_name(OUT.name + suffix + ".csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    device = str(estimator.device)
    if estimator.device.type == "cuda":
        import torch

        device += f" ({torch.cuda.get_device_name(estimator.device)})"
    w, h = read_image(FRAMES / truth[0].frame).shape[1::-1]
    agree = agreement(truth)
    pairs = [float(r["difference %"]) for r in agree if r["difference %"] != "-"]
    lines = [
        "# Counting accuracy on Kumbh footage: pretrained vs tiled / upscaled (step 3)", "",
        f"{len(truth)} frames with complete human ground truth (mean of the annotators' counts inside each frame's "
        f"counting region), shot types: {', '.join(shots[:-1])}. No model was fine-tuned and the live pipeline is "
        "unchanged: the tiled / upscaled variants exist only in `app/tiled_inference.py` and this script.", "",
        "## Pretrained vs tiled / upscaled, per shot type", "",
        "MAE in heads (relative error = sum |error| / sum true heads):", "", *table(comparison), "",
        "Bias (sum of predicted - true over sum of true heads; negative = undercount):", "", *table(bias), "",
        "## Time per frame", "", *table(timing), "",
        f"Model time per {w}x{h} frame on {device}, each model variant timed on its own over all frames (warm-up, "
        f"then the median of {REPEATS} runs per frame), in {TIMING_PASSES} separate passes; a hybrid's time is the sum "
        "of its two models' in the same pass. yes = even the slowest pass fits; borderline = the median pass fits but "
        "the slowest does not. "
        f"{CAMERAS} cameras in the {CYCLE_MS / 1e3:g} s cycle leave {CYCLE_MS / CAMERAS:.0f} ms per frame for "
        "everything; optical flow and the rest of the cycle come on top of the model time shown. Times on this laptop "
        "GPU vary between passes (power and clock states): hence the range.", "",
        "## Error per frame", "", *table(frame_rows), "",
        "## Ground truth: inter-annotator agreement", "", *table(agree), "",
        (f"Mean difference between annotators: {sum(pairs) / len(pairs):.1f}% over {len(pairs)} frames. A model "
         "error smaller than this cannot be told apart from counting noise." if pairs else
         "Single annotator: agreement not measured."), "",
        "## Notes", "",
        f"- **CSRNet**: {estimator.info['label']} Fed at 1x, 2x and 3x the frame size (bicubic upscaling; "
        f"{w}x{h} -> {2 * w}x{2 * h} -> {3 * w}x{3 * h}); density summed over the map cells whose centres lie in the "
        "region, mapped back to original pixels.",
        f"- **YOLO**: {detector.info['model']}, person class, imgsz {s.yolo_imgsz}, conf {s.yolo_conf}. Sliced: "
        f"{len(tiles(w, h, TILE, OVERLAP))} tiles of {TILE} px with {OVERLAP:.0%} overlap per {w}x{h} frame, each "
        f"upscaled to {UPSCALE_TO} px (bicubic) and detected in one batch, plus the whole frame; boxes merged across "
        "tiles when the intersection covers >= 50% of the smaller box. People count in the region by their estimated "
        "head position.",
        f"- **Hybrid**: `choose_count` with HYBRID_RULE={s.hybrid_rule}, HYBRID_THRESHOLD={s.hybrid_threshold:g} on the "
        "variant's whole-frame counts, scored on the chosen model's count in the region.",
        "- **Count error only**: the ground truth has no head positions, so localisation is not evaluated.",
        "- **Left out**: " + ("; ".join(n for n in notes if "left out" in n) or "nothing") + ".",
        "", "Which model each hybrid used:", "", *table(decisions), "",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT.with_suffix('.md')}")


if __name__ == "__main__":
    main()
