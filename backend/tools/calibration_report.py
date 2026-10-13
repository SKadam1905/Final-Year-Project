r"""How much calibration changes each zone's persons/m² and risk level, against the old estimate.

    cd backend
    ..\venv\Scripts\python.exe -m tools.calibration_report          # areas and level thresholds (no models)
    ..\venv\Scripts\python.exe -m tools.calibration_report --run    # + replay each zone's clip both ways

For every zone calibrated with tools.calibrate_zone, against the estimate it replaced (area_m2_estimated
and roi_estimated in config/zones.json):

Areas: the estimate; the calibrated polygon's exact area; and the area the live pipeline divides by,
the calibrated ground area of the density-map cells whose centres lie inside the polygon (at the
calibration's frame size and INFER_MAX_WIDTH). From these, the number of people in the zone at which
each level starts: pressure = persons/m² x (1 + k x turbulence), so with no turbulence a level starts at
threshold x area people, and turbulence lowers that.

--run: the zone's clip is read the way the live system reads a camera (a BURST_SIZE-frame burst every
BURST_INTERVAL_S s) and each burst goes through two copies of the live ZoneAnalyzer, one with the
estimate (old ROI and area) and one calibrated, sharing a single CSRNet / YOLO run. Reported: people
counted, persons/m², the 30 s pressure and its level, both ways. Pressure levels only: the optional
speed-surge trigger is left out. Runs the models: stop the demo first (scripts\stop_demo.ps1).

Writes results/zone_calibration.md, and with --run results/zone_calibration_readings.csv. Nothing is
written while no zone is calibrated.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.calibration import ZoneGeometry, ground_area_m2, zone_homography
from app.config import REPO_ROOT, Settings, ZoneConfig, get_settings, load_zones
from app.risk_engine import RiskLevel

LEVELS = (RiskLevel.YELLOW, RiskLevel.ORANGE, RiskLevel.RED)
OUT = REPO_ROOT / "results" / "zone_calibration"


def live_grid_shape(frame_size: tuple[int, int], max_width: int | None) -> tuple[int, int]:
    """(rows, cols) of the density map CSRNet gives for a (w, h) frame: 1/8 of its input size."""
    from app.csrnet_model import OUTPUT_STRIDE, stride_aligned_size

    w, h = stride_aligned_size(*frame_size, max_width)
    return h // OUTPUT_STRIDE, w // OUTPUT_STRIDE


def clip_of(zone: ZoneConfig) -> Path | None:
    """The zone's local video file (its MediaMTX clip or a file source), if there is one."""
    from app.worker import local_source

    source = local_source(zone.source)
    return Path(source) if "://" not in source and Path(source).exists() else None


def clip_frame_size(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    cap = cv2.VideoCapture(str(path))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return size if all(size) else None


def estimated_zone(zone: ZoneConfig) -> ZoneConfig:
    """The zone as it was before calibration: its old ROI (pixels of the live frames) and estimated area."""
    return ZoneConfig(zone.id, zone.name, zone.source, roi=zone.roi_estimated, area_m2=zone.area_m2_estimated,
                      graph_node=zone.graph_node, fixed_camera=zone.fixed_camera)


def roi_text(roi) -> str:
    return "whole frame" if roi is None else f"polygon of {len(roi)} corners"


def area_rows(zones: list[ZoneConfig], s: Settings) -> list[dict]:
    """Per zone: the estimated and calibrated areas, and the people at which each level starts."""
    thresholds = {RiskLevel.YELLOW: s.risk_yellow, RiskLevel.ORANGE: s.risk_orange, RiskLevel.RED: s.risk_red}
    rows = []
    for z in zones:
        row = {"zone": z.id, "area source": z.area_source,
               "estimate m²": z.area_m2_estimated if z.calibration else z.area_m2,
               "estimate ROI": roi_text(z.roi_estimated if z.calibration else z.roi)}
        if z.calibration is None:
            rows.append(row | {"calibrated m² (polygon)": "not calibrated"})
            continue
        H = zone_homography(z)
        exact = ground_area_m2(H, z.roi) if z.roi else None
        frame_size = z.frame_size or clip_frame_size(clip_of(z))
        live = (ZoneGeometry.from_zone(z).ground_area(frame_size, live_grid_shape(frame_size, s.infer_max_width))
                if frame_size else None)
        estimate = row["estimate m²"]
        source = z.calibration.source or {}
        row |= {
            "calibrated m² (polygon)": round(exact, 1) if exact is not None else "-",
            "calibrated ROI": roi_text(z.roi),
            "live m² (density grid)": round(live, 1) if live else "-",
            "estimate ÷ calibrated": round(estimate / live, 2) if estimate and live else "-",
            "measured": ", ".join(str(source[k]) for k in ("method", "measured_by", "measured_on") if source.get(k))
            or "-",
            "click spread %": source.get("area_click_sd_pct", "-"),
        }
        for level in LEVELS:
            old = f"{thresholds[level] * estimate:,.0f}" if estimate else "-"
            new = f"{thresholds[level] * live:,.0f}" if live else "-"
            row[f"people at {level} (estimate -> calibrated)"] = f"{old} -> {new}"
        rows.append(row)
    return rows


class Memo:
    """A model call remembered for the last frame, so two analyzers of one burst share a single run."""

    def __init__(self, fn):
        self.fn, self.frame, self.value = fn, None, None

    def __call__(self, frame):
        if frame is not self.frame:
            self.frame, self.value = frame, self.fn(frame)
        return self.value


def replay(zone: ZoneConfig, clip: Path, s: Settings, estimator, detector,
           max_minutes: float | None = None) -> list[dict]:
    """One row per burst of the clip: count, persons/m², 30 s pressure and level, estimated and calibrated."""
    from app.pipeline import ZoneAnalyzer
    from tools.surge_false_alarms import bursts

    s = replace(s, surge_trigger=False)
    shared_estimator = SimpleNamespace(predict=Memo(estimator.predict), info=getattr(estimator, "info", {}))
    shared_detector = SimpleNamespace(detect=Memo(detector.detect)) if detector is not None else None
    analyzers = {name: ZoneAnalyzer(z, shared_estimator, s, detector=shared_detector)
                 for name, z in (("estimated", estimated_zone(zone)), ("calibrated", zone))}
    rows = []
    for timestamp, frames in bursts(clip, s.burst_interval_s, s.burst_size, max_minutes):
        results = {name: a.analyze(frames, timestamp) for name, a in analyzers.items()}
        if any(r.scene_cut for r in results.values()):
            continue
        row = {"zone": zone.id, "t_s": round(timestamp, 2)}
        for name, r in results.items():
            st = r.status
            row |= {f"{name}_count": round(st.count, 1), f"{name}_density": round(st.density, 4),
                    f"{name}_pressure_30s": round(st.smoothed_pressure, 4), f"{name}_level": str(st.level)}
        rows.append(row)
    return rows


def summarize(zone_id: str, rows: list[dict]) -> dict:
    """Means, peaks, readings per level and level changes, estimated -> calibrated."""
    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    def share(name, level):
        return 100.0 * sum(RiskLevel(r[f"{name}_level"]) >= level for r in rows) / len(rows)

    up = sum(RiskLevel(r["calibrated_level"]) > RiskLevel(r["estimated_level"]) for r in rows)
    down = sum(RiskLevel(r["calibrated_level"]) < RiskLevel(r["estimated_level"]) for r in rows)
    old_d, new_d = mean("estimated_density"), mean("calibrated_density")
    out = {"zone": zone_id, "readings": len(rows),
           "people counted (mean)": f"{mean('estimated_count'):,.0f} -> {mean('calibrated_count'):,.0f}",
           "persons/m² (mean)": f"{old_d:.2f} -> {new_d:.2f}" + (f" (x{new_d / old_d:.2f})" if old_d > 0 else ""),
           "peak 30 s pressure": f"{max(r['estimated_pressure_30s'] for r in rows):.2f} -> "
                                 f"{max(r['calibrated_pressure_30s'] for r in rows):.2f}"}
    for level in LEVELS:
        out[f"readings at {level}+"] = f"{share('estimated', level):.0f}% -> {share('calibrated', level):.0f}%"
    out["level changed"] = f"{100.0 * (up + down) / len(rows):.0f}% ({up} up, {down} down)"
    return out


def table(rows: list[dict]) -> list[str]:
    columns = list(dict.fromkeys(k for r in rows for k in r))
    return ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns),
            *("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows)]


def main(argv=None) -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=s.zones_config)
    p.add_argument("--run", action="store_true", help="replay each calibrated zone's clip through both settings")
    p.add_argument("--zones", help="comma-separated zone ids (default: all)")
    p.add_argument("--max-minutes", type=float, help="--run: only the first N minutes of each clip")
    p.add_argument("--out", type=Path, default=OUT, help="report path without extension")
    args = p.parse_args(argv)
    zones = load_zones(args.config)
    if args.zones:
        wanted = {z.strip() for z in args.zones.split(",") if z.strip()}
        unknown = wanted - {z.id for z in zones}
        if unknown:
            raise SystemExit(f"unknown zones {sorted(unknown)}; zones: {', '.join(z.id for z in zones)}")
        zones = [z for z in zones if z.id in wanted]
    calibrated = [z for z in zones if z.calibration is not None]
    areas = area_rows(zones, s)
    print("\n".join(table(areas)))
    if not calibrated:
        print("\nNo zone is calibrated yet: run python -m tools.calibrate_zone --zone <ID> ... first. Nothing written.")
        return

    summaries, readings, notes = [], [], []
    if args.run:
        from app.csrnet_model import TRUSTED_KINDS, DensityEstimator
        from app.detector import load_detector

        estimator = DensityEstimator(s.csrnet_weights, s.device, s.infer_max_width, allow_pickle=s.csrnet_allow_pickle)
        if estimator.info["kind"] not in TRUSTED_KINDS:
            raise SystemExit(f"refusing {estimator.info['kind']} weights ({s.csrnet_weights}): the report needs "
                             "trained or imported public weights")
        detector, error = load_detector(s.yolo_weights, s.device, s.yolo_imgsz, s.yolo_conf)
        model = estimator.info["label"] + (f" + {detector.info['model']} (HYBRID_RULE={s.hybrid_rule})"
                                           if detector else f" only ({error})")
        for z in calibrated:
            clip = clip_of(z)
            if clip is None:
                notes.append(f"{z.id}: no local clip for {z.source}, not replayed.")
                continue
            if z.area_m2_estimated is None:
                notes.append(f"{z.id}: no area_m2_estimated to compare with, not replayed.")
                continue
            started = time.perf_counter()
            print(f"{z.id}: replaying {clip.name} ...", flush=True)
            rows = replay(z, clip, s, estimator, detector, args.max_minutes)
            if rows:
                summaries.append(summarize(z.id, rows))
                readings += rows
            notes.append(f"{z.id}: {len(rows)} readings from {clip.name}"
                         f"{f' (first {args.max_minutes:g} min)' if args.max_minutes else ''}, "
                         f"{time.perf_counter() - started:.0f} s.")
        print("\n".join(table(summaries)) if summaries else "nothing replayed")

    k = s.pressure_k
    lines = [
        "# Zone calibration: calibrated vs estimated ground area", "",
        f"Made by `python -m tools.calibration_report{' --run' if args.run else ''}` from {args.config.name}. "
        f"Levels: YELLOW {s.risk_yellow:g}, ORANGE {s.risk_orange:g}, RED {s.risk_red:g} on the 30 s mean of "
        f"pressure = persons/m² × (1 + {k:g} × turbulence).", "",
        "## Areas", "", *table(areas), "",
        "- *calibrated m² (polygon)*: the exact ground area of the clicked zone polygon through the homography.",
        "- *live m² (density grid)*: what the live pipeline divides by: the calibrated ground area of the "
        "density-map cells whose centres lie in the polygon (the same cells whose heads are counted).",
        "- *estimate ÷ calibrated*: for the same number of people, persons/m² is this many times higher with the "
        "calibrated area. It compares areas only: when the estimate was for another ROI (e.g. the whole frame), "
        "the people counted change as well, which only `--run` measures.",
        "- *people at LEVEL*: people in the zone at which the level starts when there is no turbulence; "
        f"turbulence T divides it by (1 + {k:g} × T).",
        "- *click spread %*: 1 sd of the calibrated area when every clicked point is off by 1 pixel "
        "(a lower bound on its uncertainty: errors in the measured width and length add to it).",
    ]
    if args.run:
        lines += ["", "## Replayed clips", "", f"Model: {model}. Speed-surge trigger left out.", ""]
        lines += [*table(summaries), "", *(f"- {n}" for n in notes)] if summaries else [*(f"- {n}" for n in notes)]
        lines += ["", "*people counted*: inside each ROI (estimated: the old ROI; calibrated: the zone polygon). "
                  "*level changed*: readings whose 30 s pressure level differs, calibrated vs estimated.",
                  "A calibration holds for the view it was made on: on a clip whose camera pans, zooms or cuts "
                  "between shots, readings from other views are not comparable."]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {args.out.with_suffix('.md')}")
    if readings:
        with open(args.out.parent / f"{args.out.name}_readings.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(readings[0]))
            writer.writeheader()
            writer.writerows(readings)
        print(f"-> {args.out.parent / f'{args.out.name}_readings.csv'}")


if __name__ == "__main__":
    main()
