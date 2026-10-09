import { describe, expect, it } from "vitest";
import { buildMapModel } from "./mapModel.js";

const NOW = 1_800_000_000_000;
const SITE = {
  nodes: [
    { id: "N", label: "North", kind: "zone", x: 100, y: 100 },
    { id: "G", label: "Ghat", kind: "ghat", x: 200, y: 100 },
    { id: "X", label: "Gate X", kind: "gate", x: 300, y: 100, exit: true },
    { id: "E", label: "East", kind: "zone", x: 100, y: 300 },
  ],
  edges: [{ a: "G", b: "N", metres: 50 }, { a: "G", b: "X", metres: 90 }, { a: "E", b: "N", metres: 80 }],
};
const T = { yellow: 2, orange: 4, red: 6 };
const zone = (id, node, level, pressure, extra = {}) => ({
  id, name: id, graphNode: node, kind: "camera", lastSeen: NOW - 1000, history: [],
  reading: { risk_level: level, pressure, count: 10, timestamp: (NOW - 1000) / 1000,
    exits: [{ exit: "X", cost: 140, path: [node, "G", "X"] }] },
  ...extra,
});

describe("buildMapModel", () => {
  it("puts zones on their nodes, takes the worst level and pressure per node", () => {
    const zones = [zone("cam1", "N", "YELLOW", 3), zone("video1", "N", "RED", 7, { kind: "video" }), zone("cam2", "Q", "GREEN", 1)];
    const m = buildMapModel({ siteMap: SITE, zones, now: NOW, connected: true, thresholds: T, selectedId: null });
    const n = m.nodes.find((x) => x.id === "N");
    expect(n.level).toBe("RED");
    expect(n.primary.id).toBe("video1");
    expect(n.pressure).toBe(7);
    expect(m.nodes.find((x) => x.id === "G").level).toBeNull(); // no camera: no level invented
    expect(m.offMap.map((z) => z.id)).toEqual(["cam2"]); // its node is not on the map
  });

  it("colours a walkway only from live pressure at its ends", () => {
    const m = buildMapModel({ siteMap: SITE, zones: [zone("cam1", "N", "ORANGE", 4.5)], now: NOW, connected: true, thresholds: T });
    const byPair = Object.fromEntries(m.edges.map((e) => [`${e.a}-${e.b}`, e]));
    expect(byPair["G-N"].band).toBe("ORANGE");
    expect(byPair["E-N"].pressure).toBe(4.5);
    expect(byPair["G-X"].band).toBeNull(); // neither end has a camera
  });

  it("draws the backend's best route for the selected zone, and nothing when its data is stale", () => {
    const zones = [zone("cam1", "N", "GREEN", 1)];
    const m = buildMapModel({ siteMap: SITE, zones, now: NOW, connected: true, thresholds: T, selectedId: "cam1" });
    expect(m.route).toEqual({ gate: "X", path: ["N", "G", "X"], points: [[100, 100], [200, 100], [300, 100]], throughTest: false });
    const stale = buildMapModel({ siteMap: SITE, zones, now: NOW, connected: false, thresholds: T, selectedId: "cam1" });
    expect(stale.route).toBeNull();
    expect(stale.nodes.find((x) => x.id === "N").level).toBe("NO SIGNAL");
  });

  it("marks test-video nodes and a live route that passes one", () => {
    const zones = [zone("cam1", "N", "GREEN", 1), zone("video1", "G", "RED", 7, { kind: "video" })];
    const m = buildMapModel({ siteMap: SITE, zones, now: NOW, connected: true, thresholds: T, selectedId: "cam1" });
    expect(m.nodes.find((x) => x.id === "G").test).toBe(true);
    expect(m.nodes.find((x) => x.id === "N").test).toBe(false);
    expect(m.route.throughTest).toBe(true); // N -> G -> X passes the test camera's node
  });
});
