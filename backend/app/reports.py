"""Reports: filtered alert and incident history, and per-zone summaries of the stored readings.

    GET /api/reports/alerts?zone_id=&level=RED&level=ORANGE&since=&until=&limit=   newest first
    GET /api/reports/incidents?zone_id=&level=&since=&until=&limit=                ORANGE/RED periods
    GET /api/reports/summary?zone_id=&since=&until=                                 per-zone peaks, time per level

Times are Unix epoch seconds. Time at a level: each reading's level holds until that zone's next
reading, but for at most `max_gap_s` (a longer gap is a camera or backend outage, counted as
"no data"); the last reading in the range holds for one burst interval. The summary covers the
readings the store still has: MongoDB keeps all of them, the in-memory store only the newest 5000.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from fastapi import FastAPI, HTTPException, Query

from app.risk_engine import RiskLevel

LEVELS = [str(level) for level in RiskLevel]


def summarise_readings(readings: list[dict], interval_s: float = 2.0, max_gap_s: float = 10.0) -> list[dict]:
    """One summary per zone from readings (each with zone_id, timestamp, count, density, pressure,
    risk_level, count_method), sorted by zone id."""
    by_zone: dict[str, list[dict]] = defaultdict(list)
    for r in readings:
        by_zone[r["zone_id"]].append(r)
    summaries = []
    for zone_id in sorted(by_zone):
        rows = sorted(by_zone[zone_id], key=lambda r: r["timestamp"])
        time_at = dict.fromkeys(LEVELS, 0.0)
        for i, r in enumerate(rows):
            span = rows[i + 1]["timestamp"] - r["timestamp"] if i + 1 < len(rows) else interval_s
            time_at[r["risk_level"]] += min(max(span, 0.0), max_gap_s)
        peak_count = max(rows, key=lambda r: r["count"])
        peak_density = max(rows, key=lambda r: r["density"])
        peak_pressure = max(rows, key=lambda r: r["pressure"])
        summaries.append({
            "zone_id": zone_id, "readings": len(rows), "first": rows[0]["timestamp"], "last": rows[-1]["timestamp"],
            "peak_count": peak_count["count"], "peak_count_time": peak_count["timestamp"],
            "peak_density": peak_density["density"], "peak_density_time": peak_density["timestamp"],
            "peak_pressure": peak_pressure["pressure"], "peak_pressure_time": peak_pressure["timestamp"],
            "highest_level": max((RiskLevel(r["risk_level"]) for r in rows), key=lambda level: level.severity),
            "time_at_level_s": {level: round(seconds, 1) for level, seconds in time_at.items()},
            "detection_readings": sum(1 for r in rows if r.get("count_method") == "detection"),
        })
    return summaries


def add_report_routes(api: FastAPI, services: Callable, settings) -> None:
    def levels_or_400(levels: list[str]) -> list[str]:
        wanted = [level.upper() for level in levels]
        unknown = sorted(set(wanted) - set(LEVELS))
        if unknown:
            raise HTTPException(400, f"unknown risk level(s) {unknown}: use {', '.join(LEVELS)}")
        return wanted

    def store_call(call):
        try:
            return call()
        except Exception as exc:  # database down
            raise HTTPException(503, f"history unavailable: {exc}") from exc

    @api.get("/api/reports/alerts")
    def report_alerts(zone_id: str | None = None, level: list[str] = Query(default=[]), since: float | None = None,
                      until: float | None = None, limit: int = 500):
        levels = levels_or_400(level)
        limit = min(max(limit, 1), 5000)
        return store_call(lambda: services().store.alert_history(zone_id, levels, since, until, limit))

    @api.get("/api/reports/incidents")
    def report_incidents(zone_id: str | None = None, level: list[str] = Query(default=[]),
                         since: float | None = None, until: float | None = None, limit: int = 500):
        levels = levels_or_400(level)
        limit = min(max(limit, 1), 5000)
        return store_call(lambda: services().store.incident_history(zone_id, levels, since, until, limit))

    @api.get("/api/reports/summary")
    def report_summary(zone_id: str | None = None, since: float | None = None, until: float | None = None):
        s = services()
        readings = store_call(lambda: s.store.readings_between(since, until, zone_id))
        names = {z.id: z.name for z in s.zones.values()}
        zones = summarise_readings(readings, settings.burst_interval_s, max(10.0, 3 * settings.burst_interval_s))
        for z in zones:
            z["name"] = names.get(z["zone_id"], z["zone_id"])
        return {"since": since, "until": until, "storage": s.store.kind, "readings": len(readings), "zones": zones}
