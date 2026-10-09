import { describe, expect, it } from "vitest";
import {
  LEVELS,
  NO_SIGNAL,
  STALE_AFTER_MS,
  WAITING,
  displayLevel,
  riskStyle,
  sortZones,
  turbulenceLabel,
  zoneState,
} from "./risk.js";

const NOW = 1_800_000_000_000;

function zone(id, level, { pressure = 1, ageMs = 0, name = id } = {}) {
  if (level === null) return { id, name, reading: null, lastSeen: null, history: [] };
  return { id, name, reading: { risk_level: level, pressure }, lastSeen: NOW - ageMs, history: [] };
}

describe("risk mapping", () => {
  it("gives every backend level a text label, description and icon", () => {
    for (const level of ["GREEN", "YELLOW", "ORANGE", "RED"]) {
      const style = riskStyle(level);
      expect(style.label).toBe(level);
      expect(style.description).toMatch(/\w/);
      expect(style.icon).toBeTruthy();
    }
    expect(new Set(Object.values(LEVELS).map((s) => s.icon)).size).toBe(4); // shapes differ too
  });

  it("ranks levels by severity", () => {
    const ranks = ["GREEN", "YELLOW", "ORANGE", "RED"].map((l) => riskStyle(l).rank);
    expect(ranks).toEqual([...ranks].sort((a, b) => a - b));
  });

  it("treats UNKNOWN and anything unexpected as waiting, not as safe", () => {
    expect(riskStyle("UNKNOWN")).toBe(WAITING);
    expect(riskStyle(undefined)).toBe(WAITING);
  });

  it("labels turbulence with the backend's bands", () => {
    expect([0, 0.29, 0.3, 0.59, 0.6, 1].map(turbulenceLabel)).toEqual(["low", "low", "medium", "medium", "high", "high"]);
  });
});

describe("stale-data rule", () => {
  it("is live up to 10 seconds, then no signal", () => {
    expect(zoneState(zone("a", "RED", { ageMs: STALE_AFTER_MS }), NOW, true)).toBe("live");
    expect(zoneState(zone("a", "RED", { ageMs: STALE_AFTER_MS + 1 }), NOW, true)).toBe("stale");
    expect(displayLevel(zone("a", "RED", { ageMs: 11_000 }), NOW, true)).toBe(NO_SIGNAL);
  });

  it("shows no signal for every zone while disconnected, even fresh ones", () => {
    expect(displayLevel(zone("a", "GREEN", { ageMs: 0 }), NOW, false)).toBe(NO_SIGNAL);
    expect(displayLevel(zone("b", null), NOW, false)).toBe(NO_SIGNAL);
  });

  it("waits for a first reading while connected", () => {
    expect(displayLevel(zone("a", null), NOW, true)).toBe(WAITING);
  });
});

describe("zone sort order", () => {
  it("puts the highest risk first, then no-signal zones above yellow", () => {
    const zones = [
      zone("green", "GREEN"),
      zone("waiting", null),
      zone("yellow", "YELLOW"),
      zone("quiet", "RED", { ageMs: 30_000 }),
      zone("red", "RED"),
      zone("orange", "ORANGE"),
    ];
    expect(sortZones(zones, NOW, true).map((z) => z.id)).toEqual(["red", "orange", "quiet", "yellow", "green", "waiting"]);
  });

  it("breaks ties by crowd pressure, then by name", () => {
    const zones = [
      zone("b", "ORANGE", { pressure: 4.2, name: "Bravo" }),
      zone("a", "ORANGE", { pressure: 4.2, name: "Alpha" }),
      zone("c", "ORANGE", { pressure: 5.9, name: "Charlie" }),
    ];
    expect(sortZones(zones, NOW, true).map((z) => z.id)).toEqual(["c", "a", "b"]);
  });

  it("does not mutate its input", () => {
    const zones = [zone("g", "GREEN"), zone("r", "RED")];
    sortZones(zones, NOW, true);
    expect(zones.map((z) => z.id)).toEqual(["g", "r"]);
  });
});
