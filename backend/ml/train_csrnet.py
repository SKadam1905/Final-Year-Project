"""Train CSRNet on processed datasets (plan §4.1).

    # ShanghaiTech Part A, starting from ImageNet VGG-16 (fits a 4 GB GPU with AMP)
    python -m ml.train_csrnet --data ../data/processed/shanghaitech_a --name parta

    # Fine-tune the Part A model on the India dataset
    python -m ml.train_csrnet --data ../data/processed/india --init ml/weights/parta_best.pth \
        --name india_ft --epochs 60 --lr 1e-6

    # Continue an interrupted run
    python -m ml.train_csrnet --data ../data/processed/shanghaitech_a --name parta --resume

Model selection uses the `val` split (carved out of train by ml.prepare_dataset), never
`test`, so the test MAE reported by ml.evaluate is a genuinely held-out number.
Outputs in --out: <name>_best.pth, <name>_last.pth (resumable) and <name>_log.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from app.config import WEIGHTS_DIR
from app.csrnet_model import CSRNet, extract_state_dict, resolve_device
from ml.dataset_index import read_index
from ml.datasets import CrowdDataset


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, action="append", required=True, help="processed dataset folder (repeatable)")
    p.add_argument("--name", default="csrnet", help="prefix for checkpoint and log files")
    p.add_argument("--out", type=Path, default=WEIGHTS_DIR)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=384, help="training crop size (multiple of 8)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10, help="epochs without val improvement before halving the LR")
    p.add_argument("--freeze-frontend-epochs", type=int, default=0, help="keep VGG-16 frozen for the first N epochs")
    p.add_argument("--init", type=Path, help="start from these weights, e.g. the Part A model for fine-tuning")
    p.add_argument("--resume", action="store_true", help="continue from <out>/<name>_last.pth")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)  # tests: skip VGG download
    p.add_argument("--max-steps", type=int, help=argparse.SUPPRESS)  # tests: cap steps per epoch
    return p.parse_args(argv)


def train_one_epoch(model, loader, optimizer, scaler, device, amp: bool, max_steps: int | None = None) -> float:
    """Mean per-image loss: sum of squared density errors, as in the CSRNet paper."""
    model.train()
    total, seen = 0.0, 0
    for step, (images, targets, _) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            preds = model(images)
        loss = F.mse_loss(preds.float(), targets, reduction="sum") / images.shape[0]
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * images.shape[0]
        seen += images.shape[0]
        if max_steps and step >= max_steps:
            break
    return total / max(seen, 1)


@torch.inference_mode()
def evaluate_counts(model, loader, device, amp: bool) -> tuple[float, float, list[float], list[float]]:
    """(MAE, RMSE, predicted counts, ground-truth counts) over full images."""
    model.eval()
    preds, gts = [], []
    for images, _, counts in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(images.to(device))
        preds.extend(out.float().sum(dim=(1, 2, 3)).cpu().tolist())
        gts.extend(counts.tolist())
    errors = np.asarray(preds) - np.asarray(gts)
    return float(np.abs(errors).mean()), float(np.sqrt((errors**2).mean())), preds, gts


def _jsonable(args: argparse.Namespace) -> dict:
    def conv(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [conv(x) for x in v]
        return v

    return {k: conv(v) for k, v in vars(args).items()}


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    amp = device.type == "cuda" and not args.no_amp

    train_samples = [s for root in args.data for s in read_index(root, ["train"])]
    val_samples = [s for root in args.data for s in read_index(root, ["val"])]
    if not train_samples or not val_samples:
        raise SystemExit("need both train and val samples; re-run ml.prepare_dataset with --val-fraction > 0")

    loader_kwargs = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                         persistent_workers=args.workers > 0)
    train_loader = DataLoader(CrowdDataset(train_samples, train=True, crop_size=args.crop),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_samples) >= args.batch_size, **loader_kwargs)
    val_loader = DataLoader(CrowdDataset(val_samples, train=False), batch_size=1, **loader_kwargs)

    model = CSRNet(pretrained_frontend=not (args.no_pretrained or args.init or args.resume))
    if args.init:
        model.load_state_dict(extract_state_dict(torch.load(args.init, map_location="cpu", weights_only=True)))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    args.out.mkdir(parents=True, exist_ok=True)
    best_path = args.out / f"{args.name}_best.pth"
    last_path = args.out / f"{args.name}_last.pth"
    log_path = args.out / f"{args.name}_log.csv"
    start_epoch, best_mae = 1, math.inf
    if args.resume:
        ckpt = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_mae = ckpt["epoch"] + 1, ckpt["best_mae"]

    print(f"train {len(train_samples)} | val {len(val_samples)} images | device {device} | amp {amp}")
    new_log = not (args.resume and log_path.exists())
    with open(log_path, "w" if new_log else "a", newline="") as log_file:
        log = csv.writer(log_file)
        if new_log:
            log.writerow(["epoch", "train_loss", "val_mae", "val_rmse", "lr", "seconds"])
        for epoch in range(start_epoch, args.epochs + 1):
            model.set_frontend_trainable(epoch > args.freeze_frontend_epochs)
            started = time.perf_counter()
            loss = train_one_epoch(model, train_loader, optimizer, scaler, device, amp, args.max_steps)
            val_mae = val_rmse = float("nan")
            if epoch % args.val_every == 0 or epoch == args.epochs:
                val_mae, val_rmse, _, _ = evaluate_counts(model, val_loader, device, amp)
                scheduler.step(val_mae)
                if val_mae < best_mae:
                    best_mae = val_mae
                    torch.save({"model": model.state_dict(), "epoch": epoch, "val_mae": val_mae,
                                "val_rmse": val_rmse, "args": _jsonable(args)}, best_path)
            seconds = time.perf_counter() - started
            lr = optimizer.param_groups[0]["lr"]
            log.writerow([epoch, f"{loss:.6f}", f"{val_mae:.4f}", f"{val_rmse:.4f}", f"{lr:.3e}", f"{seconds:.1f}"])
            log_file.flush()
            print(f"epoch {epoch:3d}/{args.epochs}  loss {loss:9.4f}  val MAE {val_mae:7.2f}  "
                  f"RMSE {val_rmse:7.2f}  best {best_mae:7.2f}  lr {lr:.1e}  {seconds:5.0f}s", flush=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "best_mae": best_mae, "args": _jsonable(args)}, last_path)
    print(f"best val MAE {best_mae:.2f} -> {best_path}")


if __name__ == "__main__":
    main()
