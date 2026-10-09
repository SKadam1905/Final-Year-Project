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
import os
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
    p.add_argument("--log", type=Path, help="per-epoch CSV log (default: <out>/<name>_log.csv)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=384, help="training crop size (multiple of 8)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10, help="epochs without val improvement before halving the LR")
    p.add_argument("--freeze-frontend-epochs", type=int, default=0, help="keep VGG-16 frozen for the first N epochs")
    p.add_argument("--init", type=Path, help="start from these weights, e.g. the Part A model for fine-tuning")
    p.add_argument("--resume", action="store_true", help="continue from <out>/<name>_last.pth")
    p.add_argument("--early-stop", type=int, default=0,
                   help="stop when the validation MAE has not improved for this many epochs (0 = never)")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-pin-memory", action="store_true",
                   help="do not page-lock batches in RAM (use on low-memory machines: it saves RAM, costs little)")
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
def evaluate_counts(model, loader, device, amp: bool) -> dict:
    """Over full images (batch 1): mae, rmse, loss (per-image sum of squared density errors,
    the training loss), and the predicted and true counts."""
    model.eval()
    preds, gts, losses = [], [], []
    for images, targets, counts in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(images.to(device))
        out = out.float()
        losses.append(F.mse_loss(out, targets.to(device), reduction="sum").item() / images.shape[0])
        preds.extend(out.sum(dim=(1, 2, 3)).cpu().tolist())
        gts.extend(counts.tolist())
    errors = np.asarray(preds) - np.asarray(gts)
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt((errors**2).mean())),
            "loss": float(np.mean(losses)), "preds": preds, "gts": gts}


def save_atomic(obj: dict, path: Path) -> None:
    """Write a checkpoint to a temporary file, then rename it: an interrupted save leaves the
    previous checkpoint intact, so resuming always works."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def provenance(args: argparse.Namespace, epoch: int, val: dict) -> dict:
    """Where the weights came from, stored inside the checkpoint (tools.inspect_weights shows it).
    test_mae/test_rmse stay empty until the one evaluation on the held-out test set fills them."""
    return {
        "source": "trained_here",
        "dataset": ", ".join(Path(d).name for d in args.data),
        "init": Path(args.init).name if args.init else "ImageNet VGG-16 front end",
        "trained": time.strftime("%Y-%m-%d"),
        "epochs": epoch,
        "val_mae": round(val["mae"], 3),
        "val_rmse": round(val["rmse"], 3),
        "test_mae": None,
        "test_rmse": None,
    }


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

    loader_kwargs = dict(num_workers=args.workers, pin_memory=device.type == "cuda" and not args.no_pin_memory,
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
    log_path = args.log or args.out / f"{args.name}_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch, best_mae, best_epoch = 1, math.inf, 0
    if args.resume:
        ckpt = torch.load(last_path, map_location="cpu", weights_only=True)
        done_epoch, done_best = ckpt["epoch"], ckpt.get("best_epoch", ckpt["epoch"])
        # Finished under the current settings? (Re-checked, so a run can be extended with more epochs.)
        if done_epoch >= args.epochs or (args.early_stop and done_epoch - done_best >= args.early_stop):
            print(f"training already finished: {ckpt.get('finished') or f'epoch {done_epoch} of {args.epochs}'}")
            return
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_mae = ckpt["epoch"] + 1, ckpt["best_mae"]
        best_epoch = ckpt.get("best_epoch", ckpt["epoch"])

    print(f"train {len(train_samples)} | val {len(val_samples)} images | device {device} | amp {amp} | "
          f"batch {args.batch_size} x {args.crop} px crops", flush=True)
    new_log = not (args.resume and log_path.exists())
    epoch_seconds = []
    with open(log_path, "w" if new_log else "a", newline="") as log_file:
        log = csv.writer(log_file)
        if new_log:
            log.writerow(["epoch", "train_loss", "val_loss", "val_mae", "val_rmse", "lr", "seconds"])
        for epoch in range(start_epoch, args.epochs + 1):
            model.set_frontend_trainable(epoch > args.freeze_frontend_epochs)
            started = time.perf_counter()
            loss = train_one_epoch(model, train_loader, optimizer, scaler, device, amp, args.max_steps)
            val = {"loss": float("nan"), "mae": float("nan"), "rmse": float("nan")}
            if epoch % args.val_every == 0 or epoch == args.epochs:
                val = evaluate_counts(model, val_loader, device, amp)
                scheduler.step(val["mae"])
                if val["mae"] < best_mae:
                    best_mae, best_epoch = val["mae"], epoch
                    save_atomic({"model": model.state_dict(), "epoch": epoch, "val_mae": val["mae"],
                                "val_rmse": val["rmse"], "args": _jsonable(args),
                                "provenance": provenance(args, epoch, val)}, best_path)
            seconds = time.perf_counter() - started
            epoch_seconds.append(seconds)
            lr = optimizer.param_groups[0]["lr"]
            log.writerow([epoch, f"{loss:.6f}", f"{val['loss']:.6f}", f"{val['mae']:.4f}", f"{val['rmse']:.4f}",
                          f"{lr:.3e}", f"{seconds:.1f}"])
            log_file.flush()
            remaining = (args.epochs - epoch) * float(np.mean(epoch_seconds))
            finish = time.strftime("%H:%M", time.localtime(time.time() + remaining))
            print(f"epoch {epoch:3d}/{args.epochs}  loss {loss:9.4f}  val loss {val['loss']:9.4f}  "
                  f"MAE {val['mae']:7.2f}  RMSE {val['rmse']:7.2f}  best {best_mae:7.2f}  lr {lr:.1e}  "
                  f"{seconds:5.0f}s  (finish ~{finish})", flush=True)
            finished = None
            if args.early_stop and epoch - best_epoch >= args.early_stop:
                finished = (f"early stop at epoch {epoch}: no better validation MAE for {args.early_stop} epochs "
                            f"(best {best_mae:.2f} at epoch {best_epoch})")
            elif epoch == args.epochs:
                finished = f"all {args.epochs} epochs done (best validation MAE {best_mae:.2f} at epoch {best_epoch})"
            save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "best_mae": best_mae, "best_epoch": best_epoch, "finished": finished,
                        "args": _jsonable(args)}, last_path)
            if finished:
                print(finished, flush=True)
                break
    print(f"best val MAE {best_mae:.2f} (epoch {best_epoch}) -> {best_path}")


if __name__ == "__main__":
    main()
