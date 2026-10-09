import { describe, expect, it } from "vitest";
import {
  MAX_POINTS,
  activeBanners,
  applyDensityUpdate,
  dismissBanner,
  initialState,
  prependAlert,
  removeZone,
  setAlerts,
  setHistory,
  setZones,
} from "./liveState.js";

const NOW = 1_800_000_000_000;

function update(zoneId, timestamp, extra = {}) {
  return { zone_id: zoneId, name: `Zone ${zoneId}`, timestamp, server_time: timestamp, risk_level: "GREEN",
    pressure: 1.5, density: 1, turbulence: 0.1, count: 100, exits: [], ...extra };
}

function alert(id, level, timestamp, zoneId = "Z1") {
  return { id, zone_id: zoneId, risk_level: level, message: `${level} in ${zoneId}`, timestamp, delivered: true,
    deliveries: [{ channel: "console", recipient: "stdout", delivered: true }] };
}

describe("density updates", () => {
  it("stores the reading and dates it by the server's clock", () => {
    const state = applyDensityUpdate(initialState, update("Z1", 1000, { server_time: 1004 }), NOW);
    const zone = state.zones.Z1;
    expect(zone.reading.pressure).toBe(1.5);
    expect(zone.lastSeen).toBe(NOW - 4000); // the reading was already 4 s old when sent
    expect(state.lastUpdate).toBe(NOW);
  });

  it("ignores the UNKNOWN placeholder of a zone with no reading yet", () => {
    const state = setZones(initialState, [{ id: "Z1", name: "North" }]);
    expect(applyDensityUpdate(state, { zone_id: "Z1", name: "North", risk_level: "UNKNOWN" }, NOW)).toBe(state);
  });

  it("keeps readings when the zone list is reloaded", () => {
    let state = applyDensityUpdate(initialState, update("Z1", 1000), NOW);
    state = setZones(state, [{ id: "Z1", name: "Renamed" }, { id: "Z2", name: "New" }]);
    expect(state.zones.Z1.reading.timestamp).toBe(1000);
    expect(state.zones.Z1.name).toBe("Renamed");
    expect(state.zones.Z2.reading).toBeNull();
  });

  it("merges history from REST and live updates by timestamp, capped", () => {
    let state = setZones(initialState, [{ id: "Z1", name: "North" }]);
    state = applyDensityUpdate(state, update("Z1", 1004), NOW);
    state = setHistory(state, "Z1", [1000, 1002, 1004].map((t) => ({ timestamp: t, pressure: t / 1000 })));
    expect(state.zones.Z1.history.map((p) => p.t)).toEqual([1000, 1002, 1004]);
    for (let t = 2000; t < 2000 + MAX_POINTS * 2; t += 2) state = applyDensityUpdate(state, update("Z1", t), NOW);
    expect(state.zones.Z1.history).toHaveLength(MAX_POINTS);
  });
});

describe("alerts", () => {
  it("prepends live alerts newest first and ignores duplicates", () => {
    let state = setAlerts(initialState, [alert(2, "ORANGE", 200), alert(1, "YELLOW", 100)]);
    state = prependAlert(state, alert(3, "ORANGE", 300));
    expect(state.alerts.map((a) => a.id)).toEqual([3, 2, 1]);
    expect(prependAlert(state, alert(3, "ORANGE", 300))).toBe(state);
  });

  it("does not lose a live alert when the REST list arrives later", () => {
    let state = prependAlert(initialState, alert(5, "ORANGE", 500));
    state = setAlerts(state, [alert(5, "ORANGE", 500), alert(4, "YELLOW", 400)]);
    expect(state.alerts.map((a) => a.id)).toEqual([5, 4]);
  });

  it("raises a banner for a live RED alert until it is dismissed", () => {
    let state = prependAlert(initialState, alert(7, "ORANGE", 700));
    expect(activeBanners(state)).toEqual([]);
    state = prependAlert(state, alert(8, "RED", 800));
    expect(activeBanners(state).map((a) => a.id)).toEqual([8]);
    state = prependAlert(state, alert(9, "RED", 900, "Z2"));
    expect(activeBanners(state).map((a) => a.id)).toEqual([9, 8]);
    state = dismissBanner(state, 8);
    expect(activeBanners(state).map((a) => a.id)).toEqual([9]);
    expect(state.alerts.map((a) => a.id)).toEqual([9, 8, 7]); // dismissing keeps the alert in the list
  });

  it("raises banners for RED alerts from REST only while they are recent", () => {
    const alerts = [alert(1, "RED", 900), alert(2, "RED", 1500), alert(3, "ORANGE", 1590)];
    const state = setAlerts(initialState, alerts, { serverTime: 1600, bannerWindowS: 600 });
    expect(activeBanners(state).map((a) => a.id)).toEqual([2]); // id 1 is 700 s old; ORANGE gets none
    expect(activeBanners(setAlerts(initialState, alerts)).map((a) => a.id)).toEqual([]);
  });

  it("keeps a dismissed banner dismissed when alerts are reloaded", () => {
    const options = { serverTime: 1600, bannerWindowS: 600 };
    let state = setAlerts(initialState, [alert(2, "RED", 1500)], options);
    state = dismissBanner(state, 2);
    state = setAlerts(state, [alert(2, "RED", 1500)], options);
    expect(activeBanners(state)).toEqual([]);
    expect(prependAlert(state, alert(2, "RED", 1500))).toBe(state);
  });

  it("keys alerts without an id by zone and time", () => {
    const unsaved = { ...alert(null, "RED", 100), id: null };
    const state = prependAlert(initialState, unsaved);
    expect(prependAlert(state, { ...unsaved })).toBe(state);
    expect(activeBanners(state)).toHaveLength(1);
  });
});

describe("video zones", () => {
  it("keeps the zone kind and forgets a removed video zone, even if a late reading arrives", () => {
    let state = setZones(initialState, [{ id: "A", name: "Camera A" }, { id: "VIDEO_1", name: "Phone video", kind: "video" }]);
    expect(state.zones.A.kind).toBe("camera");
    expect(state.zones.VIDEO_1.kind).toBe("video");
    state = removeZone(state, "VIDEO_1");
    state = applyDensityUpdate(state, update("VIDEO_1", NOW / 1000), NOW);
    expect(Object.keys(state.zones)).toEqual(["A"]);
    state = setZones(state, [{ id: "A", name: "Camera A" }, { id: "VIDEO_1", name: "Added again", kind: "video" }]);
    state = applyDensityUpdate(state, update("VIDEO_1", NOW / 1000), NOW);
    expect(state.zones.VIDEO_1.reading).not.toBeNull();
  });
});
