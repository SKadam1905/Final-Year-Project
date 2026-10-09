"""Part E: live-pipeline stress test with 1, 2, 4 and 8 cameras.

    venv\\Scripts\\python.exe scripts\\stress_test.py            # about 25 minutes
    venv\\Scripts\\python.exe scripts\\stress_test.py --seconds 20 --cameras 1 2   # quick check

Builds the real services (app.worker.build_services, FRAME_SOURCE=local: every camera is a
reader thread decoding the test clip at its own frame rate) and drives the same cycle the
server runs (app.main.run_cycle) every BURST_INTERVAL_S, for each configuration:
    flow 640 px  (before the Part A speed-up)
    flow 320 px  (after)
    flow 320 px + INFER_MAX_WIDTH 960   (smaller CSRNet input, for comparison)
The CSRNet weights are untrained (random): the network is the same size, so timing is
representative, but risk levels are meaningless. Alerts go to the console only.
Writes results/stress_test.csv/.md and results/stress_test.png.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ["ALERT_CHANNEL"] = "console"  # never send real Telegram messages from a test
from app.config import get_settings  # noqa: E402
from app.csrnet_model import CSRNet  # noqa: E402
from app.main import run_cycle  # noqa: E402
from app.worker import build_services  # noqa: E402
from report_utils import save_figure, write_table  # noqa: E402

NODES = ["RAMKUND_NORTH", "RAMKUND_SOUTH", "RAMKUND_EAST", "RAMKUND_WEST", "GHAT_1", "GHAT_2", "GHAT_3", "RAMKUND_NORTH"]
CONFIGS = [("before: flow 640", 640, 1280), ("after: flow 320", 320, 1280), ("after: flow 320, CSRNet 960", 320, 960)]


def random_weights(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(0)
        torch.save({"model": CSRNet(pretrained_frontend=False).state_dict()}, path)
    return path


def zones_file(directory: Path, cameras: int, clip: Path) -> Path:
    zones = [{"id": f"CAM_{i + 1}", "name": f"Camera {i + 1}", "source": str(clip), "area_m2": 5000.0,
              "graph_node": NODES[i % len(NODES)]} for i in range(cameras)]
    path = directory / f"zones_{cameras}.json"
    path.write_text(json.dumps({"zones": zones}), encoding="utf-8")
    return path


async def measure(services, seconds: float, interval: float) -> dict:
    worker = services.worker
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and any(services.source.peek_burst(z) is None for z in worker.zones):
        await asyncio.sleep(0.1)  # wait for every camera's first burst

    async def no_emit(events):
        no_emit.updates.extend(p["zone_id"] for name, p in events if name == "density_update")

    no_emit.updates = []
    for _ in range(3):  # warm-up: CUDA kernels, cuDNN autotuning
        await run_cycle(worker, no_emit)
        await asyncio.sleep(interval)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    cycles, last_update, gaps = [], {}, []
    cpu0, wall0 = time.process_time(), time.monotonic()
    while time.monotonic() - wall0 < seconds:
        started = time.monotonic()
        no_emit.updates = []
        await run_cycle(worker, no_emit)
        duration = time.monotonic() - started
        now = time.monotonic()
        for zone in set(no_emit.updates):
            if zone in last_update:
                gaps.append(now - last_update[zone])
            last_update[zone] = now
        cycles.append((duration, len(set(no_emit.updates))))
        await asyncio.sleep(max(0.0, interval - duration))
    wall, cpu = time.monotonic() - wall0, time.process_time() - cpu0

    durations = [d for d, _ in cycles]
    zones = len(worker.zones)
    return {
        "cycles": len(cycles),
        "mean_latency_s": round(statistics.mean(durations), 3),
        "p95_latency_s": round(float(np.percentile(durations, 95)), 3),
        "max_latency_s": round(max(durations), 3),
        "late_cycles_pct": round(100 * sum(d > interval for d in durations) / len(cycles), 1),
        "all_zones_updated_pct": round(100 * sum(n == zones for _, n in cycles) / len(cycles), 1),
        "missed_zone_updates": sum(zones - n for _, n in cycles),
        "max_update_gap_s": round(max(gaps), 2) if gaps else "",
        "gpu_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0,
        "gpu_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20) if torch.cuda.is_available() else 0,
        "cpu_pct_of_one_core": round(100 * cpu / wall),
        "cpu_pct_of_machine": round(100 * cpu / wall / os.cpu_count(), 1),
    }


def meets(row: dict, interval: float) -> str:
    """Within the cycle: p95 latency under the interval and at most 1 % of cycles late.

    Missed camera updates are reported but not counted here: with a looping test clip each camera
    skips one update per loop (its reader drops the buffer at the wrap so a burst never mixes the
    clip's end and start), which a live camera would not do.
    """
    return "yes" if float(row["p95_latency_s"]) <= interval and float(row["late_cycles_pct"]) <= 1 else "no"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", type=Path, default=ROOT / "data" / "test_clips" / "synthetic_crowd_1280.mp4")
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--replot", action="store_true", help="re-render the table and figure from the existing CSV")
    args = p.parse_args(argv)
    base = get_settings()
    if args.replot:
        import csv

        with open(args.out / "stress_test.csv", newline="") as f:
            rows = [{k: (int(v) if k in ("cameras", "flow_width_px", "csrnet_width_px") else v) for k, v in r.items()}
                    for r in csv.DictReader(f)]
        for row in rows:
            row["meets_2s_cycle"] = meets(row, base.burst_interval_s)
        report(rows, args, base)
        return
    if not args.clip.exists():
        raise SystemExit(f"{args.clip} missing; run scripts\\make_test_clip.py first")

    weights = random_weights(ROOT / "data" / "test_clips" / "untrained_csrnet.pth")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for label, flow_width, infer_width in CONFIGS:
            for cameras in args.cameras:
                if infer_width != 1280 and cameras < 4:
                    continue  # the smaller-input comparison only matters where 1280 struggles
                settings = replace(base, frame_source="local", mongodb_uri="", alert_channels=("console",),
                                   csrnet_weights=weights, zones_config=zones_file(Path(tmp), cameras, args.clip),
                                   flow_max_width=flow_width, infer_max_width=infer_width)
                services = build_services(settings)
                services.source.start()
                try:
                    result = asyncio.run(measure(services, args.seconds, settings.burst_interval_s))
                finally:
                    services.source.stop()
                row = {"config": label, "flow_width_px": flow_width, "csrnet_width_px": infer_width,
                       "cameras": cameras, **result}
                row["meets_2s_cycle"] = meets(row, settings.burst_interval_s)
                rows.append(row)
                print(row, flush=True)
                del services
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    report(rows, args, base)


def report(rows: list[dict], args, base) -> None:
    def max_cameras(label):
        ok = [int(r["cameras"]) for r in rows if r["config"] == label and r["meets_2s_cycle"] == "yes"]
        return max(ok) if ok else 0

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    args.cameras = sorted({int(r["cameras"]) for r in rows})
    notes = [f"{gpu}; {os.cpu_count()} logical CPUs; clip {args.clip.name} (1280x720, 25 fps) for every camera; "
             f"{args.seconds:.0f} s per row after 3 warm-up cycles; cycle = app.main.run_cycle every "
             f"{base.burst_interval_s:.0f} s.",
             "Latency = time to process every camera once (CSRNet on the GPU, one frame at a time; optical flow on "
             "the CPU, cameras in parallel). A cycle is late when it takes longer than the 2 s interval. "
             "'meets_2s_cycle' = p95 latency <= 2 s and at most 1 % of cycles late.",
             "missed_zone_updates: the test clip is 64 s long and loops; at each wrap a camera's reader drops its "
             "buffer (so no burst mixes the clip's end and start) and that camera skips one update, about one miss "
             "per camera per 64 s. A live camera does not loop.",
             "max_update_gap_s = longest time between two updates of one camera (the dashboard shows 'No signal' "
             "after 10 s). GPU memory is PyTorch's peak; the CUDA context adds about 0.3-0.5 GB.",
             "Untrained CSRNet weights: timing is representative (same network), risk levels are not.",
             *(f"Maximum cameras within the 2 s cycle, {label}: {max_cameras(label)}" for label, _, _ in CONFIGS
               if any(r["config"] == label for r in rows))]
    write_table(args.out / "stress_test", rows, "Live-pipeline stress test", notes)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (label, _, _), colour, marker in zip(CONFIGS, ["#b91c1c", "#0f766e", "#1d4ed8"], "osD"):
        data = sorted((int(r["cameras"]), float(r["mean_latency_s"]), float(r["p95_latency_s"]))
                      for r in rows if r["config"] == label)
        if not data:
            continue
        cams, mean, p95 = zip(*data)
        ax.plot(cams, mean, marker=marker, color=colour, label=f"{label} (mean)")
        ax.plot(cams, p95, marker=marker, color=colour, linestyle=":", alpha=0.8, label=f"{label} (p95)")
    ax.axhline(base.burst_interval_s, color="black", linestyle="--", linewidth=1.2, label="2 s cycle deadline")
    ax.set_xscale("log", base=2)
    ax.set_xticks(args.cameras, [str(c) for c in args.cameras])
    ax.set_xlabel("Cameras (count)")
    ax.set_ylabel("Cycle latency (s)")
    ax.set_title("Time to process every camera once")
    ax.legend(fontsize=8)
    save_figure(fig, args.out / "stress_test.png")
    for label, _, _ in CONFIGS:
        if any(r["config"] == label for r in rows):
            print(f"max cameras within 2 s, {label}: {max_cameras(label)}")


if __name__ == "__main__":
    main()
