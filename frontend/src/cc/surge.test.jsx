import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { surgeRuleText, surgeStatus } from "../risk.js";
import AlertFeed from "./AlertFeed.jsx";
import ZoneList from "./ZoneList.jsx";
import ZoneSummary from "./ZoneSummary.jsx";

const NOW = 1_800_000_000_000;
const OFF = { trigger: false, ratio: 1.1, consecutive: 2, history_s: 10, min_level: "YELLOW" };
const ON = { ...OFF, trigger: true };
const reading = (extra) => ({
  zone_id: "GHAT_1", risk_level: "YELLOW", count: 250, density: 2.5, turbulence: 0.1, pressure: 2.6,
  timestamp: (NOW - 1000) / 1000, surge_ratio: 1.02, surge_fired: false, escalation: null, ...extra,
});
const zone = (extra) => ({ id: "GHAT_1", name: "Ghat 1 approach", kind: "camera", lastSeen: NOW - 1000, history: [],
  reading: reading(extra) });

describe("surgeStatus", () => {
  it("says what the ratio means, in words as well as colour", () => {
    expect(surgeStatus(reading({ surge_ratio: null }), OFF)).toMatchObject({ text: "—", tone: "idle" });
    expect(surgeStatus(reading({}), OFF)).toMatchObject({ text: "1.02×", tone: "normal", note: "usual pace" });
    expect(surgeStatus(reading({ surge_ratio: 1.3 }), OFF)).toMatchObject({ tone: "rising" });
    expect(surgeStatus(reading({ surge_ratio: 1.6, surge_fired: true }), OFF).note).toMatch(/trigger off/);
    expect(surgeStatus(reading({ surge_ratio: 1.6, surge_fired: true }), ON)).toMatchObject({ tone: "fired", note: "speed surge" });
    expect(surgeStatus(reading({ surge_ratio: 1.6, surge_fired: true, escalation: "speed surge" }), ON).tone).toBe("raised");
  });

  it("describes the trigger setting", () => {
    expect(surgeRuleText(OFF)).toBe("Trigger off (would fire at ≥ 1.1× for 2 readings in a row).");
    expect(surgeRuleText(ON)).toMatch(/^Trigger on: .* raises a YELLOW\+ zone to ORANGE\.$/);
  });
});

describe("speed-surge indicator", () => {
  it("shows every live zone's ratio in the zone list, even with the trigger off", () => {
    render(<ZoneList zones={[zone({ surge_ratio: 1.45, surge_fired: true })]} now={NOW} connected selectedId={null}
      onSelect={vi.fn()} surge={OFF} />);
    const indicator = screen.getByText(/surge 1\.45×/);
    expect(indicator.textContent).toContain("⚡");
    expect(indicator.getAttribute("title")).toMatch(/trigger off/);
  });

  it("shows the selected zone's ratio and when the trigger raised the level", () => {
    render(<ZoneSummary zone={zone({ risk_level: "ORANGE", surge_ratio: 1.6, surge_fired: true, escalation: "speed surge" })}
      now={NOW} connected surge={ON} />);
    expect(screen.getByText("1.60× ⚡")).toBeTruthy();
    expect(screen.getByText(/speed surge: level raised/)).toBeTruthy();
  });

  it("marks alerts raised by a speed surge", () => {
    const alert = { id: 7, zone_id: "GHAT_1", risk_level: "ORANGE", message: "ORANGE: Ghat 1 approach - SPEED SURGE",
      timestamp: NOW / 1000 - 5, delivered: true, deliveries: [], hotspot: null, gate_actions: [], reason: "speed surge",
      acknowledged_by: null, acknowledged_at: null };
    render(<AlertFeed alerts={[alert, { ...alert, id: 8, reason: null }]} zoneNames={{ GHAT_1: "Ghat 1 approach" }}
      onAcknowledge={vi.fn()} />);
    expect(screen.getAllByText("⚡ speed surge")).toHaveLength(1);
  });
});
