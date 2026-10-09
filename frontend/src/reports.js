// Pure helpers for the Reports page: query strings, time ranges and CSV export.

export const LEVELS = ["GREEN", "YELLOW", "ORANGE", "RED"];

export const PRESETS = [
  { value: "1h", label: "Last hour", seconds: 3600 },
  { value: "6h", label: "Last 6 hours", seconds: 6 * 3600 },
  { value: "24h", label: "Last 24 hours", seconds: 24 * 3600 },
  { value: "7d", label: "Last 7 days", seconds: 7 * 24 * 3600 },
  { value: "custom", label: "Custom range" },
];

/** { zoneId, levels, since, until } -> "?zone_id=..&level=RED&level=ORANGE&since=..&until=.." (empty parts left out). */
export function reportQuery({ zoneId = "", levels = [], since = null, until = null }, withLevels = true) {
  const params = new URLSearchParams();
  if (zoneId) params.set("zone_id", zoneId);
  if (withLevels) levels.forEach((level) => params.append("level", level));
  if (since != null) params.set("since", String(since));
  if (until != null) params.set("until", String(until));
  const query = params.toString();
  return query ? `?${query}` : "";
}

/** "2026-10-09T14:05" (a datetime-local value, local time) -> epoch seconds, or null. */
export function localToEpoch(value) {
  if (!value) return null;
  const ms = new Date(value).getTime();
  return Number.isFinite(ms) ? Math.round(ms / 1000) : null;
}

/** Epoch ms -> a datetime-local value in local time. */
export function epochToLocal(ms) {
  const d = new Date(ms);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** The [since, until] epoch seconds a preset or custom range stands for. */
export function timeRange(preset, from, to, nowMs = Date.now()) {
  const found = PRESETS.find((p) => p.value === preset);
  if (found?.seconds) {
    const until = Math.round(nowMs / 1000);
    return { since: until - found.seconds, until };
  }
  return { since: localToEpoch(from), until: localToEpoch(to) };
}

/** Seconds as "1 h 05 min", "12 min 30 s" or "45 s". */
export function formatSpan(seconds) {
  const s = Math.round(seconds ?? 0);
  if (s >= 3600) return `${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")} min`;
  if (s >= 60) return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
  return `${s} s`;
}

function csvCell(value) {
  if (value == null) return "";
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** Rows to CSV text; `columns` are [{ key, label, value?(row) }]. */
export function toCsv(rows, columns) {
  const header = columns.map((c) => csvCell(c.label)).join(",");
  const lines = rows.map((row) => columns.map((c) => csvCell(c.value ? c.value(row) : row[c.key])).join(","));
  return [header, ...lines].join("\r\n") + "\r\n";
}

/** Save CSV text as a file through a temporary link. The UTF-8 byte-order mark makes Excel read it as
 * UTF-8; without it Excel uses the ANSI code page and garbles names such as "रामकुंड घाट". */
export function downloadCsv(filename, csv) {
  const url = URL.createObjectURL(new Blob(["﻿", csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const isoTime = (epochSeconds) => (epochSeconds == null ? "" : new Date(epochSeconds * 1000).toISOString());

export const ALERT_COLUMNS = [
  { key: "timestamp", label: "time_utc", value: (a) => isoTime(a.timestamp) },
  { key: "zone_id", label: "zone_id" },
  { key: "risk_level", label: "risk_level" },
  { key: "delivered", label: "delivered" },
  { key: "message", label: "message" },
];

export const SUMMARY_COLUMNS = [
  { key: "zone_id", label: "zone_id" },
  { key: "name", label: "zone" },
  { key: "readings", label: "readings" },
  { key: "first", label: "first_utc", value: (z) => isoTime(z.first) },
  { key: "last", label: "last_utc", value: (z) => isoTime(z.last) },
  { key: "peak_count", label: "peak_count", value: (z) => z.peak_count?.toFixed(1) },
  { key: "peak_count_time", label: "peak_count_time_utc", value: (z) => isoTime(z.peak_count_time) },
  { key: "peak_density", label: "peak_density_per_m2", value: (z) => z.peak_density?.toFixed(3) },
  { key: "peak_density_time", label: "peak_density_time_utc", value: (z) => isoTime(z.peak_density_time) },
  { key: "peak_pressure", label: "peak_pressure", value: (z) => z.peak_pressure?.toFixed(3) },
  { key: "highest_level", label: "highest_level" },
  ...LEVELS.map((level) => ({ key: level, label: `time_${level.toLowerCase()}_s`, value: (z) => z.time_at_level_s?.[level] })),
  { key: "detection_readings", label: "readings_counted_by_yolo" },
];

export const INCIDENT_COLUMNS = [
  { key: "zone_id", label: "zone_id" },
  { key: "started_at", label: "started_utc" },
  { key: "ended_at", label: "ended_utc" },
  { key: "peak_level", label: "peak_level" },
  { key: "peak_pressure", label: "peak_pressure", value: (i) => i.peak_pressure?.toFixed(3) },
  { key: "peak_density", label: "peak_density_per_m2", value: (i) => i.peak_density?.toFixed(3) },
  { key: "peak_count", label: "peak_count", value: (i) => i.peak_count?.toFixed(1) },
];
