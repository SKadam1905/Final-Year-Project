// The Command Centre's KPI strip, computed only from live backend data (zones from Socket.IO,
// alerts, /health). A zone counts as live while its reading is under 10 s old (risk.js).

import { displayLevel, sortZones, zoneState } from "../risk.js";

export function computeKpis({ zones, alerts, health, now, connected }) {
  const live = zones.filter((z) => zoneState(z, now, connected) === "live");
  const byLevel = { RED: 0, ORANGE: 0, YELLOW: 0, GREEN: 0, "NO SIGNAL": 0, WAITING: 0 };
  for (const zone of zones) byLevel[displayLevel(zone, now, connected).label] += 1;
  const top = sortZones(zones, now, connected)[0];
  const topLevel = top ? displayLevel(top, now, connected).label : null;
  const cameras = health?.cameras ? Object.values(health.cameras) : [];
  return {
    people: live.length ? live.reduce((sum, z) => sum + Math.max(0, z.reading.count ?? 0), 0) : null,
    liveZones: live.length,
    testZones: live.filter((z) => z.kind === "video").length,
    byLevel,
    highest: top && ["RED", "ORANGE", "YELLOW", "GREEN"].includes(topLevel) ? { level: topLevel, zone: top } : null,
    unacknowledged: alerts.filter((a) => !a.acknowledged_at).length,
    camerasOnline: cameras.filter((c) => c?.status === "online").length,
    camerasTotal: Math.max(cameras.length, zones.length),
  };
}
