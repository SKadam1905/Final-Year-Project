"""Training curves from results/train_log.csv (written by scripts/train_full.ps1).

    venv\\Scripts\\python.exe scripts\\plot_training.py        # -> results\\training_curve.png
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from report_utils import save_figure

ROOT = Path(__file__).resolve().parents[1]
PAPER_MAE = 68.2  # Li et al., CVPR 2018, ShanghaiTech Part A test set


def number(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, default=ROOT / "results" / "train_log.csv")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "training_curve.png")
    args = p.parse_args(argv)
    with open(args.log, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.log} has no epochs yet")
    col = {key: [number(r[key]) for r in rows] for key in rows[0]}
    epochs = col["epoch"]

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(epochs, col["train_loss"], label="train (384 px crops)")
    if "val_loss" in col:
        ax1.plot(epochs, col["val_loss"], label="validation (full images)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss: sum of squared density error per image (people², log scale)")
    ax1.set_title("(a) Loss")
    ax1.legend()

    ax2.plot(epochs, col["val_mae"], label="validation MAE", color="#0f766e")
    ax2.plot(epochs, col["val_rmse"], label="validation RMSE", color="#475569", alpha=0.7)
    ax2.axhline(PAPER_MAE, color="#b91c1c", linestyle="--", label=f"paper test MAE {PAPER_MAE}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Count error (people per image)")
    ax2.set_title("(b) Validation counting error")
    lr_axis = ax2.twinx()
    lr_axis.plot(epochs, col["lr"], color="#a16207", linestyle=":", label="learning rate")
    lr_axis.set_yscale("log")
    lr_axis.set_ylabel("Learning rate (log scale)")
    lines = ax2.get_legend_handles_labels()
    lr_lines = lr_axis.get_legend_handles_labels()
    ax2.legend(lines[0] + lr_lines[0], lines[1] + lr_lines[1], fontsize=8)
    save_figure(fig, args.out)
    best = min(range(len(rows)), key=lambda i: col["val_mae"][i] if not math.isnan(col["val_mae"][i]) else math.inf)
    print(f"{len(rows)} epochs; best validation MAE {col['val_mae'][best]:.2f} at epoch {int(epochs[best])} -> {args.out}")


if __name__ == "__main__":
    main()
