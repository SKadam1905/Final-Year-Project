import { describe, expect, it } from "vitest";
import { ALERT_COLUMNS, formatSpan, reportQuery, timeRange, toCsv } from "./reports.js";

describe("reports helpers", () => {
  it("builds the query string, repeating level", () => {
    expect(reportQuery({ zoneId: "GHAT_1", levels: ["RED", "ORANGE"], since: 10, until: 20 }))
      .toBe("?zone_id=GHAT_1&level=RED&level=ORANGE&since=10&until=20");
    expect(reportQuery({ levels: ["RED"], since: 10 }, false)).toBe("?since=10");
    expect(reportQuery({})).toBe("");
  });

  it("turns a preset into the last N seconds and a custom range into epoch seconds", () => {
    expect(timeRange("1h", "", "", 7_200_000)).toEqual({ since: 3600, until: 7200 });
    const custom = timeRange("custom", "2026-10-09T10:00", "2026-10-09T11:30");
    expect(custom.until - custom.since).toBe(5400);
    expect(timeRange("custom", "", "")).toEqual({ since: null, until: null });
  });

  it("formats spans of time", () => {
    expect([formatSpan(45), formatSpan(750), formatSpan(3900), formatSpan(null)]).toEqual(["45 s", "12 min 30 s", "1 h 05 min", "0 s"]);
  });

  it("writes CSV with quoting for commas, quotes and line breaks", () => {
    const csv = toCsv([{ timestamp: 0, zone_id: "GHAT_1", risk_level: "RED", delivered: true,
      message: 'RED: Ghat 1, "crush"\nGate actions:' }], ALERT_COLUMNS);
    expect(csv).toBe('time_utc,zone_id,risk_level,delivered,message\r\n'
      + '1970-01-01T00:00:00.000Z,GHAT_1,RED,true,"RED: Ghat 1, ""crush""\nGate actions:"\r\n');
  });
});
