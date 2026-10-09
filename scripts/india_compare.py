"""India fine-tuning: compare the model before and after on the India test videos.

    venv\\Scripts\\python.exe scripts\\india_compare.py results\\india_before\\results.csv results\\india_after\\results.csv \\
        --index data\\processed\\india\\index.csv --weights backend\\ml\\weights\\india_ft_best.pth

Reads the per-image CSVs written by ml.evaluate for the same test split (scripts\\india_finetune.ps1
runs both) and writes results\\india_finetune.csv/.md: MAE and RMSE before and after, overall and
per test video. With --weights it also stores the after-numbers in that checkpoint's provenance.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from report_utils import write_table  # noqa: E402


def read_results(path: Path) -> dict[str, tuple[float, float]]:
    with open(path, newline="") as f:
        return {r["image_name"]: (float(r["gt_count"]), float(r["pred_count"])) for r in csv.DictReader(f)}


def errors(rows: dict[str, tuple[float, float]], names: list[str]) -> tuple[float, float]:
    e = [rows[n][1] - rows[n][0] for n in names]
    return sum(abs(x) for x in e) / len(e), math.sqrt(sum(x * x for x in e) / len(e))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("--index", type=Path, help="the processed dataset's index.csv, for per-video numbers")
    p.add_argument("--weights", type=Path, help="fine-tuned checkpoint: store the test numbers in its provenance")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "india_finetune")
    args = p.parse_args(argv)

    before, after = read_results(args.before), read_results(args.after)
    if set(before) != set(after):
        raise SystemExit("the two evaluations did not use the same test images")
    video_of = {}
    if args.index:
        with open(args.index, newline="") as f:
            for r in csv.DictReader(f):
                video_of[r["id"]] = r["video"]  # ml.evaluate names each image by its index id
    groups = {"all test videos": sorted(before)}
    for name in sorted(before):
        if video_of.get(name):
            groups.setdefault(video_of[name], []).append(name)

    rows = []
    for group, names in groups.items():
        mae_b, rmse_b = errors(before, names)
        mae_a, rmse_a = errors(after, names)
        rows.append({"test_set": group, "images": len(names), "mae_before": round(mae_b, 2), "mae_after": round(mae_a, 2),
                     "rmse_before": round(rmse_b, 2), "rmse_after": round(rmse_a, 2),
                     "mae_change_pct": round(100 * (mae_a - mae_b) / mae_b, 1) if mae_b else ""})
    write_table(args.out, rows, "India fine-tuning: counting error on the held-out India test videos", [
        "Before = the starting weights (Part A); after = fine-tuned on the India training videos. Same test images, "
        "evaluated once each; the test videos were never used for training or for choosing the epoch.",
        "Plan target: 5-10% lower MAE after fine-tuning.",
    ])
    if args.weights:
        import torch

        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
        checkpoint.setdefault("provenance", {}).update(test_mae=rows[0]["mae_after"], test_rmse=rows[0]["rmse_after"],
                                                        test_set="India test videos")
        torch.save(checkpoint, args.weights)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
