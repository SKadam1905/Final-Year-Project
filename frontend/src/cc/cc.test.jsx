import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import KpiStrip from "./KpiStrip.jsx";
import { computeKpis } from "./kpi.js";
import { modelName } from "./theme.js";
import TopBar from "./TopBar.jsx";

const NOW = 1_800_000_000_000;
const zone = (id, level, count, pressure, kind = "camera", ageMs = 1000) => ({
  id, name: id, kind, lastSeen: NOW - ageMs, history: [],
  reading: { zone_id: id, risk_level: level, count, density: 1, turbulence: 0.1, pressure, timestamp: (NOW - ageMs) / 1000 },
});

describe("computeKpis", () => {
  it("counts only live zones, groups levels, finds the highest risk and the unacknowledged alerts", () => {
    const zones = [zone("A", "GREEN", 100.4, 0.5), zone("B", "RED", 300, 6.5), zone("C", "ORANGE", 50, 4.2, "video"),
      zone("D", "GREEN", 999, 0.1, "camera", 30_000)]; // D: 30 s old -> no signal, not counted
    const alerts = [{ id: 1, acknowledged_at: null }, { id: 2, acknowledged_at: 5 }, { id: 3 }];
    const health = { cameras: { A: { status: "online" }, B: { status: "online" }, C: { status: "online" }, D: { status: "offline" } } };
    const k = computeKpis({ zones, alerts, health, now: NOW, connected: true });
    expect(k.people).toBeCloseTo(450.4);
    expect(k.liveZones).toBe(3);
    expect(k.testZones).toBe(1);
    expect(k.byLevel).toMatchObject({ RED: 1, ORANGE: 1, GREEN: 1, "NO SIGNAL": 1 });
    expect(k.highest.level).toBe("RED");
    expect(k.highest.zone.id).toBe("B");
    expect(k.unacknowledged).toBe(2);
    expect([k.camerasOnline, k.camerasTotal]).toEqual([3, 4]);
  });

  it("has no people figure and no highest risk while disconnected", () => {
    const k = computeKpis({ zones: [zone("A", "RED", 300, 7)], alerts: [], health: null, now: NOW, connected: false });
    expect(k.people).toBeNull();
    expect(k.highest).toBeNull();
    expect(k.byLevel["NO SIGNAL"]).toBe(1);
  });
});

describe("KpiStrip", () => {
  it("shows a dash, not a made-up number, when there is no data", () => {
    const k = computeKpis({ zones: [], alerts: [], health: null, now: NOW, connected: false });
    render(<KpiStrip kpis={k} health={null} />);
    expect(screen.getByText("no live camera")).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(3);
  });

  it("names the highest-risk zone with the level word", () => {
    const k = computeKpis({ zones: [zone("Ghat 1", "RED", 300, 7)], alerts: [], health: null, now: NOW, connected: true });
    render(<KpiStrip kpis={k} health={null} />);
    expect(screen.getByText("Ghat 1")).toBeInTheDocument();
    expect(screen.getAllByText("RED").length).toBeGreaterThan(0);
  });
});

describe("modelName", () => {
  it("names the weights and the detector", () => {
    expect(modelName({ model: { kind: "public_pretrained", dataset: "ShanghaiTech Part A" }, detector: { model: "yolov8s" } }))
      .toBe("CSRNet · public weights (ShanghaiTech Part A) + YOLOv8s");
    expect(modelName({ model_loaded: false, model: null })).toBe("No trained model");
  });
});

describe("TopBar", () => {
  const props = { connection: "live", now: NOW, onLogout: vi.fn(), sound: null, page: "live" };

  it("shows admin pages only to the admin, and logs out", async () => {
    const onLogout = vi.fn();
    const { rerender } = render(<TopBar {...props} onLogout={onLogout} user={{ name: "Officer", role: "officer", role_label: "Security Officer" }} />);
    const nav = () => screen.getByRole("navigation", { name: "Views" }).textContent;
    expect(nav()).toContain("Command Centre");
    expect(nav()).toContain("Classic view");
    expect(nav()).not.toContain("Video analysis");
    expect(nav()).not.toContain("Reports");
    rerender(<TopBar {...props} onLogout={onLogout} user={{ name: "Admin", role: "admin", role_label: "Control Room Admin" }} />);
    expect(nav()).toContain("Video analysis");
    expect(nav()).toContain("Reports");
    await userEvent.click(screen.getByRole("button", { name: "Log out" }));
    expect(onLogout).toHaveBeenCalled();
  });

  it("says LIVE or DISCONNECTED in words", () => {
    const user = { name: "A", role: "admin", role_label: "Control Room Admin" };
    const { rerender } = render(<TopBar {...props} user={user} />);
    expect(screen.getByRole("status")).toHaveTextContent("LIVE");
    rerender(<TopBar {...props} user={user} connection="disconnected" />);
    expect(screen.getByRole("status")).toHaveTextContent("DISCONNECTED");
  });
});
