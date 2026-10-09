import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { api } from "../api.js";
import GhatMap from "./GhatMap.jsx";

const NOW = 1_800_000_000_000;
const SITE = {
  schematic: true, note: "Schematic map, not to scale", width: 1000, height: 600,
  nodes: [
    { id: "RAMKUND_NORTH", label: "Ramkund North", kind: "zone", x: 650, y: 205, exit: false },
    { id: "GHAT_1", label: "Ghat 1", kind: "ghat", x: 530, y: 150, exit: false },
    { id: "EXIT_GATE_A", label: "Gate A", kind: "gate", x: 480, y: 55, exit: true },
  ],
  edges: [{ a: "GHAT_1", b: "RAMKUND_NORTH", metres: 50 }, { a: "EXIT_GATE_A", b: "GHAT_1", metres: 90 }],
  unplaced: [],
};
const ZONES = [{
  id: "RAMKUND_NORTH", name: "Ramkund north steps", graphNode: "RAMKUND_NORTH", kind: "camera", lastSeen: NOW - 500, history: [],
  reading: { risk_level: "RED", count: 812, density: 6.5, turbulence: 0.4, pressure: 7.1, timestamp: (NOW - 500) / 1000,
    exits: [{ exit: "EXIT_GATE_A", cost: 900, path: ["RAMKUND_NORTH", "GHAT_1", "EXIT_GATE_A"] }] },
}];

describe("GhatMap", () => {
  it("labels itself schematic, shows the live level and count, selects on click, and routes to the safest gate", async () => {
    vi.spyOn(api, "siteMap").mockResolvedValue(SITE);
    const onSelect = vi.fn();
    const { container } = render(<GhatMap zones={ZONES} now={NOW} connected thresholds={{ yellow: 2, orange: 4, red: 6 }}
      selectedId="RAMKUND_NORTH" onSelect={onSelect} />);
    expect(await screen.findByText("Schematic map, not to scale")).toBeInTheDocument();
    const node = screen.getByRole("button", { name: /Ramkund North: RED, 812 people/ });
    expect(within(node).getByText("■ RED")).toBeInTheDocument();
    await userEvent.click(node);
    expect(onSelect).toHaveBeenCalledWith("RAMKUND_NORTH");
    expect(container.querySelectorAll("polyline")).toHaveLength(2); // glow + animated dash
    expect(screen.getByText("SAFEST EXIT")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Ghat 1, no camera" })).toBeInTheDocument();
  });

  it("explains a missing map instead of showing nothing", async () => {
    vi.spyOn(api, "siteMap").mockRejectedValue(new Error("site map unavailable"));
    render(<GhatMap zones={[]} now={NOW} connected thresholds={null} selectedId={null} onSelect={() => {}} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Map unavailable: site map unavailable");
  });
});
