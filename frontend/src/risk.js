// Risk levels exactly as the backend sends them (backend/app/risk_engine.py):
// GREEN < YELLOW < ORANGE < RED, plus "UNKNOWN" for a zone with no reading yet.
// Colour is never the only cue: every state has a text label, a short description and an
// icon. Badge colours keep text contrast at WCAG AA or better (dark text on yellow/orange,
// white on green/red/grey), which also holds up on a phone in sunlight.

export const STALE_AFTER_MS = 10_000;

export const LEVELS = {
  RED: {
    label: "RED", description: "Danger", icon: "■", rank: 5,
    badge: "bg-red-700 text-white ring-red-950", border: "border-red-700",
  },
  ORANGE: {
    label: "ORANGE", description: "Warning", icon: "◆", rank: 4,
    badge: "bg-orange-500 text-black ring-orange-900", border: "border-orange-500",
  },
  YELLOW: {
    label: "YELLOW", description: "Watch", icon: "▲", rank: 2,
    badge: "bg-yellow-400 text-black ring-yellow-800", border: "border-yellow-400",
  },
  GREEN: {
    label: "GREEN", description: "Normal", icon: "●", rank: 1,
    badge: "bg-green-700 text-white ring-green-950", border: "border-green-700",
  },
};

// A zone that has gone quiet ranks above YELLOW: it may be in trouble with nobody watching.
export const NO_SIGNAL = {
  label: "NO SIGNAL", description: "No recent data", icon: "✕", rank: 3,
  badge: "bg-gray-700 text-white ring-gray-900", border: "border-gray-400",
};

export const WAITING = {
  label: "WAITING", description: "No reading yet", icon: "…", rank: 0,
  badge: "bg-white text-gray-900 ring-gray-500", border: "border-gray-300",
};

export function riskStyle(level) {
  return LEVELS[level] ?? WAITING;
}

/** Same bands as backend/ml/optical_flow.py turbulence_to_label. */
export function turbulenceLabel(score) {
  if (score >= 0.6) return "high";
  if (score >= 0.3) return "medium";
  return "low";
}

/**
 * "live": a reading under 10 s old while connected; "stale": older, or the socket is down
 * (never show an old level as current); "waiting": connected but no reading yet.
 */
export function zoneState(zone, now, connected) {
  if (!connected) return "stale";
  if (!zone.reading) return "waiting";
  return now - zone.lastSeen > STALE_AFTER_MS ? "stale" : "live";
}

export function displayLevel(zone, now, connected) {
  const state = zoneState(zone, now, connected);
  if (state === "live") return riskStyle(zone.reading.risk_level);
  return state === "stale" ? NO_SIGNAL : WAITING;
}

/** Highest risk first; then higher crowd pressure; then by name. */
export function sortZones(zones, now, connected) {
  const key = (zone) => {
    const level = displayLevel(zone, now, connected);
    const live = zoneState(zone, now, connected) === "live";
    return { rank: level.rank, pressure: live ? zone.reading.pressure : -1 };
  };
  return [...zones].sort((a, b) => {
    const ka = key(a);
    const kb = key(b);
    return kb.rank - ka.rank || kb.pressure - ka.pressure || a.name.localeCompare(b.name);
  });
}
