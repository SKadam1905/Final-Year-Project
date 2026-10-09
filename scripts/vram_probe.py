"""Part B: find the largest training batch and crop that fit in this GPU's memory.

    venv\\Scripts\\python.exe scripts\\vram_probe.py        # -> results\\training_vram.csv/.md

Runs real CSRNet training steps (mixed precision, Adam, the training loss) on random tensors for
each batch x crop setting and records PyTorch's peak memory and the time per step. A setting
"fits" if it runs and its peak reserved memory leaves at least 0.5 GB of the card free
(the CUDA context and Windows' display use the rest).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.csrnet_model import CSRNet  # noqa: E402
from report_utils import write_table  # noqa: E402

SETTINGS = [(8, 256), (8, 384), (8, 448), (8, 512), (12, 384), (16, 384), (4, 512), (6, 512)]
HEADROOM_GB = 0.5


def probe(model, optimizer, scaler, batch: int, crop: int, steps: int = 5) -> tuple[float, float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, 3, crop, crop, device="cuda")
    y = torch.rand(batch, 1, crop // 8, crop // 8, device="cuda")
    times = []
    for i in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        loss = F.mse_loss(out.float(), y, reduction="sum") / batch
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        if i:
            times.append(time.perf_counter() - t0)
    return (torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30,
            sum(times) / len(times))


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    model = CSRNet(pretrained_frontend=False).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    rows = []
    for batch, crop in SETTINGS:
        try:
            allocated, reserved, step = probe(model, optimizer, scaler, batch, crop)
            fits = reserved <= total - HEADROOM_GB
            rows.append({"batch": batch, "crop_px": crop, "pixels_per_step_M": round(batch * crop * crop / 1e6, 2),
                         "peak_allocated_gb": round(allocated, 2), "peak_reserved_gb": round(reserved, 2),
                         "s_per_step": round(step, 3), "fits": "yes" if fits else "no (too close to the limit)"})
        except torch.OutOfMemoryError:
            rows.append({"batch": batch, "crop_px": crop, "pixels_per_step_M": round(batch * crop * crop / 1e6, 2),
                         "peak_allocated_gb": "", "peak_reserved_gb": "", "s_per_step": "", "fits": "no (out of memory)"})
        print(rows[-1], flush=True)
    fitting = [r for r in rows if r["fits"] == "yes"]
    largest = max(fitting, key=lambda r: (r["pixels_per_step_M"], r["crop_px"])) if fitting else None
    write_table(ROOT / "results" / "training_vram", rows, "CSRNet training memory by batch and crop size", [
        f"{torch.cuda.get_device_name(0)}, {total:.1f} GB; mixed precision (torch.amp), Adam, batch of random tensors.",
        f"'fits' = runs and leaves >= {HEADROOM_GB} GB free for the CUDA context and the display.",
        f"Largest setting that fits: batch {largest['batch']} x {largest['crop_px']} px." if largest else "Nothing fits.",
    ])


if __name__ == "__main__":
    main()
