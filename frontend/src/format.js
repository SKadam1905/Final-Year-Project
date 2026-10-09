const clock = new Intl.DateTimeFormat("en-IN", {
  hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
});

/** HH:MM:SS for a time in milliseconds. */
export const formatTime = (ms) => (ms == null ? "—" : clock.format(new Date(ms)));

export const formatCount = (n) => Math.round(Math.max(0, n)).toLocaleString("en-IN");

export const formatNumber = (n, digits = 2) => Math.max(0, n).toFixed(digits);

/** Seconds of video as m:ss (or h:mm:ss), e.g. 75.4 -> "1:15". */
export function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

/** EXIT_GATE_C -> "Exit gate C", RAMKUND_NORTH -> "Ramkund north". */
export function prettyNode(node) {
  const words = node.toLowerCase().split("_");
  return words
    .map((w, i) => (w.length === 1 ? w.toUpperCase() : i === 0 ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

/** Which model produced a count (backend count_method). */
export const COUNT_METHODS = { detection: "YOLO detection", density: "CSRNet density map" };
export const countMethodLabel = (method) => COUNT_METHODS[method] ?? COUNT_METHODS.density;

/** "top-right of the view: ~24 people (2.40 /m²)" for a backend hotspot, or null. */
export function hotspotText(hotspot) {
  if (!hotspot) return null;
  return `${hotspot.label} of the view: ~${formatCount(hotspot.count)} people (${formatNumber(hotspot.density)} /m²)`;
}
