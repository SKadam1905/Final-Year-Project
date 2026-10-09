import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import ZoneCard from "./ZoneCard.jsx";

const NOW = 1_800_000_000_000;

function zone(level, ageMs = 1000) {
  return {
    id: "RAMKUND_NORTH",
    name: "Ramkund north steps",
    lastSeen: NOW - ageMs,
    history: [],
    reading: {
      zone_id: "RAMKUND_NORTH", risk_level: level, count: 1234.4, density: 5.123, turbulence: 0.45, pressure: 6.27,
      timestamp: (NOW - ageMs) / 1000,
      exits: [{ exit: "EXIT_GATE_C", cost: 120, path: ["RAMKUND_NORTH", "GHAT_1", "EXIT_GATE_C"] }],
    },
  };
}

describe("ZoneCard", () => {
  it.each([
    ["GREEN", "Normal"],
    ["YELLOW", "Watch"],
    ["ORANGE", "Warning"],
    ["RED", "Danger"],
  ])("shows %s as text with its readings", (level, description) => {
    render(<ZoneCard zone={zone(level)} now={NOW} connected onSelect={() => {}} />);
    expect(screen.getByText(level)).toBeInTheDocument();
    expect(screen.getByText(description)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Ramkund north steps" })).toBeInTheDocument();
    expect(screen.getByText("1,234")).toBeInTheDocument();
    expect(screen.getByText("5.12 /m²")).toBeInTheDocument();
    expect(screen.getByText("0.45 (medium)")).toBeInTheDocument();
    expect(screen.getByText("6.27")).toBeInTheDocument();
    expect(screen.getByText("1. Exit gate C")).toBeInTheDocument();
    expect(screen.getByText(/Ramkund north → Ghat 1 → Exit gate C/)).toBeInTheDocument();
  });

  it("greys out a stale zone and never shows its old level", () => {
    render(<ZoneCard zone={zone("RED", 11_000)} now={NOW} connected onSelect={() => {}} />);
    expect(screen.getByText("NO SIGNAL")).toBeInTheDocument();
    expect(screen.queryByText("RED")).not.toBeInTheDocument();
    expect(screen.queryByText("1,234")).not.toBeInTheDocument();
    expect(screen.getByText(/No signal since .* Risk level unknown/)).toBeInTheDocument();
    expect(screen.getByRole("article")).toHaveClass("bg-gray-200");
  });

  it("shows no signal for a fresh reading while disconnected", () => {
    render(<ZoneCard zone={zone("GREEN", 0)} now={NOW} connected={false} onSelect={() => {}} />);
    expect(screen.getByText("NO SIGNAL")).toBeInTheDocument();
    expect(screen.queryByText("GREEN")).not.toBeInTheDocument();
  });

  it("waits for the first reading", () => {
    render(<ZoneCard zone={{ ...zone("GREEN"), reading: null, lastSeen: null }} now={NOW} connected onSelect={() => {}} />);
    expect(screen.getByText("WAITING")).toBeInTheDocument();
    expect(screen.getByText("Waiting for the first reading…")).toBeInTheDocument();
  });

  it("opens the zone when clicked", async () => {
    const onSelect = vi.fn();
    render(<ZoneCard zone={zone("ORANGE")} now={NOW} connected onSelect={onSelect} />);
    await userEvent.click(screen.getByRole("button", { name: "Ramkund north steps" }));
    expect(onSelect).toHaveBeenCalledWith("RAMKUND_NORTH");
  });

  it("says which model counted and where the hotspot is", () => {
    const z = zone("ORANGE");
    z.reading = { ...z.reading, count_method: "density", detection_count: 7, density_count: 1234.4,
      hotspot: { row: 0, col: 2, label: "top-right", count: 310.2, density: 6.25 } };
    render(<ZoneCard zone={z} now={NOW} connected onSelect={() => {}} />);
    expect(screen.getByText("CSRNet density map")).toBeInTheDocument();
    expect(screen.getByText(/YOLO found 7/)).toBeInTheDocument();
    expect(screen.getByText(/top-right of the view: ~310 people \(6.25 \/m²\)/)).toBeInTheDocument();
  });

  it("shows a detection count without the YOLO aside", () => {
    const z = zone("GREEN");
    z.reading = { ...z.reading, count: 12, count_method: "detection", detection_count: 12, density_count: 30 };
    render(<ZoneCard zone={z} now={NOW} connected onSelect={() => {}} />);
    expect(screen.getByText("YOLO detection")).toBeInTheDocument();
    expect(screen.queryByText(/YOLO found/)).not.toBeInTheDocument();
  });

  it("lists the gate actions for the zone's level", () => {
    const z = zone("RED");
    z.reading = { ...z.reading, gate_actions: [
      { kind: "stop_inflow", text: "STOP inflow at all gates feeding this zone: Gate C, Gate A.", gates: [] },
      { kind: "open_all_exits", text: "Open all exits: Gate A, Gate B, Gate C, Gate D.", gates: [] },
    ] };
    render(<ZoneCard zone={z} now={NOW} connected onSelect={() => {}} />);
    expect(screen.getByRole("heading", { name: "Gate actions" })).toBeInTheDocument();
    expect(screen.getByText("STOP inflow at all gates feeding this zone: Gate C, Gate A.")).toBeInTheDocument();
    expect(screen.getByText("Open all exits: Gate A, Gate B, Gate C, Gate D.")).toBeInTheDocument();
  });
});
