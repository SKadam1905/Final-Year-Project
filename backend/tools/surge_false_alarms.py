r"""Speed-surge false alarms per hour on a long video of normal crowd behaviour (to tune SURGE_RATIO).

    cd backend
    ..\venv\Scripts\python.exe -m tools.surge_false_alarms ..\data\india_dataset\raw_videos\ghat\long_normal.mp4
    ..\venv\Scripts\python.exe -m tools.surge_false_alarms <video> --ratios 1.1,1.2,1.5 --zone RAMKUND_NORTH

The video is read the way the live system reads a camera: one burst of BURST_SIZE consecutive frames
every BURST_INTERVAL_S seconds of video. Each burst goes through the production turbulence step
(ml.optical_flow.burst_turbulence, live FlowParams: FLOW_MAX_WIDTH, camera-motion compensation), whose
mean speed feeds one app.surge.SurgeTracker per candidate ratio (SURGE_CONSECUTIVE readings in a row,
SURGE_HISTORY_S of history). A burst spanning a scene cut is skipped and resets the trackers, as live;
a cut falling between two bursts is not seen (live neither), so edited footage (news clips) can show
false surges at shot changes that a fixed camera never has: prefer long, uncut recordings.

On footage known to show no surge, every firing is a false alarm. For each ratio the tool reports how
often the rule starts firing (switch-ons) per hour of video, and the share of readings on which it fires.
Without --zone this is the rule alone, before the density gate: an upper bound. With --zone ZONE_ID (its
ROI, area and fixed_camera flag from config/zones.json) CSRNet runs as well, with the weights in .env, and
the gate is applied as live: a firing counts only while the zone's pressure level is SURGE_MIN_LEVEL or
above. Writes results/surge_false_alarms/<video>.md and .csv unless --no-report.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import cv2

from app.config import REPO_ROOT, get_settings, load_zones
from app.risk_engine import RiskLevel
from app.surge import SurgeTracker
from ml.event_metrics import switch_ons
from ml.optical_flow import FlowParams, burst_has_cut, burst_turbulence

DEFAULT_RATIOS = (1.05, 1.10, 1.15, 1.20, 1.30, 1.50, 2.00)


def bursts(video: Path, interval_s: float, size: int, max_minutes: float | None = None):
    """(timestamp in seconds, [frames]) every interval_s of video: `size` consecutive frames each."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(size, round(interval_s * fps))
    last = round(max_minutes * 60 * fps) if max_minutes else None
    index, frames = 0, []
    while last is None or index < last:
        if index % step < size:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if len(frames) == size:
                yield (index - size + 1) / fps, frames
                frames = []
        elif not cap.grab():  # frames between bursts: decoded, not converted
            break
        index += 1
    cap.release()


def measure(video: Path, ratios, consecutive: int, history_s: float, interval_s: float, burst_size: int,
            flow: FlowParams, analyzer=None, min_level: RiskLevel = RiskLevel.YELLOW,
            max_minutes: float | None = None) -> tuple[list[dict], dict]:
    """Per ratio: readings, readings fired, switch-ons (and gated ones with an analyzer), per hour."""
    trackers = {r: SurgeTracker(r, consecutive, history_s) for r in ratios}
    fired = {r: [] for r in ratios}
    gated = {r: [] for r in ratios}
    readings = cuts = 0
    first = last = None
    started = time.perf_counter()
    for timestamp, frames in bursts(video, interval_s, burst_size, max_minutes):
        first = timestamp if first is None else first
        last = timestamp
        if analyzer is not None:
            analysis = analyzer.analyze(frames, timestamp)
            if analysis.scene_cut:
                cuts += 1
                for t in trackers.values():
                    t.reset()
                continue
            speed, gate = analysis.status.speed, analysis.status.level >= min_level
        else:
            if burst_has_cut(frames, flow):
                cuts += 1
                for t in trackers.values():
                    t.reset()
                continue
            speed, gate = burst_turbulence(frames, None, flow).mean_speed, True
        readings += 1
        for r, t in trackers.items():
            on = t.update(speed, timestamp).fired
            fired[r].append(on)
            gated[r].append(on and gate)
        if readings % 300 == 0:
            print(f"  {timestamp / 60:.1f} min of video, {time.perf_counter() - started:.0f} s", flush=True)
    hours = ((last - first) + interval_s) / 3600 if readings else 0.0
    rows = []
    for r in ratios:
        row = {"ratio": r, "readings": readings,
               "readings fired": f"{100 * sum(fired[r]) / readings:.1f}%" if readings else "-",
               "false alarms (switch-ons)": switch_ons(fired[r]),
               "per hour, rule alone": f"{switch_ons(fired[r]) / hours:.1f}" if hours else "-"}
        if analyzer is not None:
            row[f"per hour, with the {min_level}+ density gate"] = f"{switch_ons(gated[r]) / hours:.1f}" if hours else "-"
        rows.append(row)
    return rows, {"hours": hours, "readings": readings, "scene_cuts": cuts, "seconds": time.perf_counter() - started}


def main(argv=None) -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path, help="a long video of normal crowd behaviour (no surge in it)")
    p.add_argument("--ratios", default=",".join(map(str, DEFAULT_RATIOS)), help="comma-separated SURGE_RATIO values")
    p.add_argument("--consecutive", type=int, default=s.surge_consecutive)
    p.add_argument("--history-s", type=float, default=s.surge_history_s)
    p.add_argument("--interval-s", type=float, default=s.burst_interval_s)
    p.add_argument("--zone", help="apply the density gate with this zone's calibration (runs CSRNet)")
    p.add_argument("--max-minutes", type=float, help="only the first N minutes of the video")
    p.add_argument("--no-report", action="store_true", help="print only; do not write results/surge_false_alarms/")
    args = p.parse_args(argv)
    ratios = sorted({float(r) for r in args.ratios.split(",") if r.strip()} | {s.surge_ratio})
    min_level = RiskLevel(s.surge_min_level)
    analyzer, zone_note = None, "rule alone (no density gate): an upper bound"
    flow = FlowParams(max_width=s.flow_max_width)
    if args.zone:
        from app.csrnet_model import TRUSTED_KINDS, DensityEstimator
        from app.pipeline import ZoneAnalyzer

        zones = {z.id: z for z in load_zones(s.zones_config)}
        if args.zone not in zones:
            raise SystemExit(f"unknown zone {args.zone}; zones: {', '.join(zones)}")
        estimator = DensityEstimator(s.csrnet_weights, s.device, s.infer_max_width, allow_pickle=s.csrnet_allow_pickle)
        if estimator.info["kind"] not in TRUSTED_KINDS:
            raise SystemExit(f"refusing {estimator.info['kind']} weights for the density gate ({s.csrnet_weights})")
        analyzer = ZoneAnalyzer(zones[args.zone], estimator, replace(s, surge_trigger=False))
        zone_note = (f"zone {args.zone}: its ROI and area; density gate {min_level}+ with {estimator.info['label']}")
    print(f"{args.video.name}: one {s.burst_size}-frame burst every {args.interval_s:g} s; {zone_note}")
    rows, info = measure(args.video, ratios, args.consecutive, args.history_s, args.interval_s, s.burst_size, flow,
                         analyzer, min_level, args.max_minutes)
    columns = list(rows[0]) if rows else []
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(str(row[c]) for c in columns))
    print(f"{info['readings']} readings over {info['hours'] * 60:.1f} min of video, {info['scene_cuts']} scene cuts "
          f"skipped, {info['seconds']:.0f} s to compute")
    if args.no_report or not rows:
        return
    out = REPO_ROOT / "results" / "surge_false_alarms" / args.video.stem
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# Speed-surge false alarms: {args.video.name}", "",
             f"{info['readings']} readings (one {s.burst_size}-frame burst every {args.interval_s:g} s) over "
             f"{info['hours'] * 60:.1f} min of video assumed to contain no surge; {info['scene_cuts']} bursts with a "
             f"scene cut skipped. Rule: ratio reached on {args.consecutive} readings in a row, {args.history_s:g} s of "
             f"history; {zone_note}.", "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns),
             *("| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows), "",
             "False alarms = how often the rule starts firing (a firing that lasts several readings counts once).",
             f"Current SURGE_RATIO: {s.surge_ratio} (trigger {'on' if s.surge_trigger else 'off'})."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
