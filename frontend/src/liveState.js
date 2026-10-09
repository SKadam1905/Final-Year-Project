// Pure state transitions for the dashboard (used by useLiveZones through useReducer).
// Zones are keyed by their backend id; a zone looks like
//   { id, name, kind: "camera" | "video", reading: <last density_update payload> | null, lastSeen: ms | null,
//     history: [{ t, pressure }] }
// "video" zones are uploaded test videos looping as cameras; they can be removed at runtime.
// Alerts are the backend's alert events (same shape over REST and Socket.IO), newest first.

export const MAX_ALERTS = 50;
export const MAX_POINTS = 300; // per zone: 10 minutes at one reading every 2 s

export const initialState = { zones: {}, alerts: [], banners: [], dismissed: [], removed: [], lastUpdate: null };

function emptyZone(id, name) {
  return { id, name: name ?? id, reading: null, lastSeen: null, history: [] };
}

/** Zone list from GET /api/zones; readings already received are kept. */
export function setZones(state, configs) {
  const zones = {};
  for (const config of configs) {
    zones[config.id] = { ...(state.zones[config.id] ?? emptyZone(config.id)), name: config.name, kind: config.kind ?? "camera" };
  }
  return { ...state, zones, removed: state.removed.filter((id) => !zones[id]) };
}

/** A video zone was removed: drop it, and ignore readings for it still on their way. */
export function removeZone(state, zoneId) {
  const zones = { ...state.zones };
  delete zones[zoneId];
  return { ...state, zones, removed: [...state.removed, zoneId] };
}

/** When the reading was taken, on this browser's clock (server_time gives its age). */
export function seenAt(payload, now) {
  const ageSeconds = payload.server_time != null ? Math.max(0, payload.server_time - payload.timestamp) : 0;
  return now - ageSeconds * 1000;
}

export function mergeHistory(existing, points) {
  const byTime = new Map(existing.map((p) => [p.t, p]));
  for (const p of points) byTime.set(p.t, p);
  return [...byTime.values()].sort((a, b) => a.t - b.t).slice(-MAX_POINTS);
}

/** A "density_update" event (or GET /status body) received at `now` (ms). */
export function applyDensityUpdate(state, payload, now) {
  if (payload.risk_level === "UNKNOWN" || payload.timestamp == null) return state; // no reading yet
  if (state.removed.includes(payload.zone_id)) return state;
  const zone = state.zones[payload.zone_id] ?? emptyZone(payload.zone_id, payload.name);
  const updated = {
    ...zone,
    reading: payload,
    lastSeen: seenAt(payload, now),
    history: mergeHistory(zone.history, [{ t: payload.timestamp, pressure: payload.pressure }]),
  };
  return { ...state, lastUpdate: now, zones: { ...state.zones, [payload.zone_id]: updated } };
}

/** Readings from GET /api/zones/{id}/history, merged into the chart data. */
export function setHistory(state, zoneId, readings) {
  const zone = state.zones[zoneId];
  if (!zone) return state;
  const points = readings.map((r) => ({ t: r.timestamp, pressure: r.pressure }));
  return { ...state, zones: { ...state.zones, [zoneId]: { ...zone, history: mergeHistory(zone.history, points) } } };
}

export function alertKey(alert) {
  return alert.id ?? `${alert.zone_id}@${alert.timestamp}`;
}

function newestFirst(alerts) {
  const seen = new Set();
  return alerts
    .filter((a) => !seen.has(alertKey(a)) && seen.add(alertKey(a)))
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, MAX_ALERTS);
}

function withBanner(banners, alert, dismissed) {
  const key = alertKey(alert);
  return alert.risk_level !== "RED" || banners.includes(key) || dismissed.includes(key) ? banners : [key, ...banners];
}

/**
 * GET /api/alerts/recent, merged with alerts that arrived live meanwhile. RED alerts younger
 * than `bannerWindowS` (by the server's clock) also get a banner: the backend will not repeat
 * an alert within its cooldown, so a guard opening the page during a RED would otherwise never
 * see one.
 */
export function setAlerts(state, alerts, { serverTime = null, bannerWindowS = 0 } = {}) {
  let banners = state.banners;
  if (serverTime != null) {
    for (const alert of alerts) {
      if (serverTime - alert.timestamp <= bannerWindowS) banners = withBanner(banners, alert, state.dismissed);
    }
  }
  return { ...state, alerts: newestFirst([...state.alerts, ...alerts]), banners };
}

/** A live "alert" event: prepended; a RED one also raises a banner until dismissed. */
export function prependAlert(state, alert) {
  if (state.alerts.some((a) => alertKey(a) === alertKey(alert))) return state;
  return { ...state, alerts: newestFirst([alert, ...state.alerts]), banners: withBanner(state.banners, alert, state.dismissed) };
}

/** Dismissed banners stay dismissed, even when the alert list is reloaded. */
export function dismissBanner(state, key) {
  return { ...state, banners: state.banners.filter((k) => k !== key), dismissed: [...state.dismissed, key] };
}

/** The alerts whose banners are still showing, newest first. */
export function activeBanners(state) {
  return state.banners
    .map((key) => state.alerts.find((a) => alertKey(a) === key))
    .filter(Boolean);
}

export function reducer(state, action) {
  switch (action.type) {
    case "zones":
      return setZones(state, action.zones);
    case "density":
      return applyDensityUpdate(state, action.payload, action.now);
    case "history":
      return setHistory(state, action.zoneId, action.readings);
    case "alerts":
      return setAlerts(state, action.alerts, action);
    case "alert":
      return prependAlert(state, action.alert);
    case "dismiss":
      return dismissBanner(state, action.key);
    case "removeZone":
      return removeZone(state, action.zoneId);
    default:
      return state;
  }
}
