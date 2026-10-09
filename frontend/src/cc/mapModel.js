// Turns the schematic site map (GET /api/site-map) and the live zones into what the map draws.
// Nothing is invented: a node shows a level only when a camera zone sits on it; a walkway is
// coloured only from live crowd pressure at its ends; the route is the backend's exits[0].path.

import { displayLevel, zoneState } from "../risk.js";
import { bandFor, levelUi } from "./theme.js";

export function buildMapModel({ siteMap, zones, now, connected, thresholds, selectedId }) {
  if (!siteMap) return null;
  const byId = Object.fromEntries(siteMap.nodes.map((n) => [n.id, { ...n, zones: [] }]));
  const offMap = [];
  for (const zone of zones) {
    if (zone.graphNode && byId[zone.graphNode]) byId[zone.graphNode].zones.push(zone);
    else offMap.push(zone);
  }

  const nodes = Object.values(byId).map((node) => {
    if (!node.zones.length) return { ...node, primary: null, level: null, pressure: null, live: false };
    const ranked = [...node.zones].sort((a, b) =>
      levelUi(displayLevel(b, now, connected).label).rank - levelUi(displayLevel(a, now, connected).label).rank);
    const primary = ranked[0];
    const liveZones = node.zones.filter((z) => zoneState(z, now, connected) === "live");
    return {
      ...node,
      primary,
      test: node.zones.every((z) => z.kind === "video"), // only test videos here: never part of live routing
      level: displayLevel(primary, now, connected).label,
      live: liveZones.length > 0,
      pressure: liveZones.length ? Math.max(...liveZones.map((z) => z.reading.pressure)) : null,
    };
  });
  const placed = Object.fromEntries(nodes.map((n) => [n.id, n]));

  const edges = siteMap.edges.map((edge) => {
    const known = [placed[edge.a].pressure, placed[edge.b].pressure].filter((p) => p != null);
    const pressure = known.length ? Math.max(...known) : null;
    return { ...edge, from: placed[edge.a], to: placed[edge.b], pressure, band: pressure == null ? null : bandFor(pressure, thresholds) };
  });

  let route = null;
  const selected = zones.find((z) => z.id === selectedId);
  const best = selected && zoneState(selected, now, connected) === "live" ? selected.reading.exits?.[0] : null;
  if (best && best.path.every((id) => placed[id])) {
    route = {
      gate: best.exit, path: best.path, points: best.path.map((id) => [placed[id].x, placed[id].y]),
      throughTest: selected.kind !== "video" && best.path.some((id) => placed[id].test),
    };
  }
  return { nodes, edges, route, offMap, selectedNode: selected?.graphNode ?? null };
}
