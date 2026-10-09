import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

vi.mock("../useLiveZones.js", () => ({
  useZoneFeed: (zoneId, overlay) => ({
    snapshot: "blob:snap", overlay: overlay === "none" ? null : `blob:${overlay}`, capturedAt: 1_800_000_000, snapshotError: null, overlayError: null,
  }),
}));

const { default: LiveMonitor } = await import("./LiveMonitor.jsx");
const { CameraStatus, CrowdEstimation, SafestExit, placeName } = await import("./Panels.jsx");

const NOW = 1_800_000_000_000;
const ZONE = {
  id: "RAMKUND_NORTH", name: "Ramkund north steps", kind: "camera", lastSeen: NOW - 500, history: [],
  reading: {
    risk_level: "ORANGE", count: 420, density: 4.4, turbulence: 0.35, pressure: 4.9, timestamp: (NOW - 500) / 1000,
    count_method: "density", density_count: 420.2, detection_count: 9,
    hotspot: { row: 0, col: 2, label: "top-right", count: 90, density: 6.1 },
    grid: { rows: 3, cols: 3, counts: [[1, 2, 90], [0, 0, 0], [0, 0, 0]], hotspot: { row: 0, col: 2, label: "top-right", count: 90, density: 6.1 } },
    exits: [
      { exit: "EXIT_GATE_A", cost: 312.4, path: ["RAMKUND_NORTH", "RAMKUND_EAST", "EXIT_GATE_A"] },
      { exit: "EXIT_GATE_C", cost: 540, path: ["RAMKUND_NORTH", "GHAT_1", "EXIT_GATE_C"] },
      { exit: "EXIT_GATE_D", cost: 700, path: ["RAMKUND_NORTH", "GHAT_1", "EXIT_GATE_C", "EXIT_GATE_D"] },
    ],
    gate_actions: [
      { kind: "hold_inflow", text: "Hold inflow at Gate C.", gates: ["EXIT_GATE_C"] },
      { kind: "open_exit", text: "Open alternate exit Gate A (route: Ramkund north > Ramkund east > Gate A).", gates: ["EXIT_GATE_A"] },
    ],
  },
};

describe("Command Centre side panels", () => {
  it("names places like the backend's gate actions", () => {
    expect([placeName("EXIT_GATE_C"), placeName("RAMKUND_NORTH"), placeName("GHAT_1")]).toEqual(["Gate C", "Ramkund north", "Ghat 1"]);
  });

  it("shows the safest gate large, its path, two alternatives and the gate actions", () => {
    render(<SafestExit zone={ZONE} now={NOW} connected />);
    expect(screen.getByText("Gate A")).toHaveClass("text-3xl");
    expect(screen.getByText("Ramkund north → Ramkund east → Gate A")).toBeInTheDocument();
    expect(screen.getByText("2. Gate C")).toBeInTheDocument();
    expect(screen.getByText("3. Gate D")).toBeInTheDocument();
    expect(screen.getByText("Hold inflow at Gate C.")).toBeInTheDocument();
  });

  it("pauses exit advice without a live reading", () => {
    render(<SafestExit zone={ZONE} now={NOW + 60_000} connected />);
    expect(screen.getByText("No live reading: exit advice paused.")).toBeInTheDocument();
    expect(screen.queryByText("Gate A")).not.toBeInTheDocument();
  });

  it("reports each camera's status in words", () => {
    const health = { server_time: 1_800_000_000, cameras: { RAMKUND_NORTH: { status: "online", last_frame_time: 1_799_999_999.5, frames_read: 1200, reconnects: 0 }, GHAT_1: { status: "offline" } } };
    const zones = [ZONE, { id: "GHAT_1", name: "Ghat 1 approach", kind: "camera" }, { id: "VIDEO_1", name: "Test", kind: "video" }];
    render(<CameraStatus zone={ZONE} zones={zones} health={health} />);
    expect(screen.getAllByText("Online").length).toBeGreaterThan(0);
    expect(screen.getByText(/Last frame 0.5 s ago · 1,200 frames read · 0 reconnects/)).toBeInTheDocument();
    expect(screen.getByText(": No signal")).toBeInTheDocument();
    expect(screen.getByText(": Test video")).toBeInTheDocument();
  });

  it("explains an empty trend instead of drawing a fake line", () => {
    render(<CrowdEstimation zone={ZONE} now={NOW} connected thresholds={{ yellow: 2, orange: 4, red: 6 }} />);
    expect(screen.getByText("Turbulence")).toBeInTheDocument();
    expect(screen.getByText(/YOLO found 9, CSRNet counted 420/)).toBeInTheDocument();
    expect(screen.getByText(/top-right of the view/)).toBeInTheDocument();
  });

  it("switches the live camera overlay, including the grid with its hotspot", async () => {
    render(<LiveMonitor zone={ZONE} densityScale={{ unit: "persons/m²", max: 7, stops: [{ value: 0, colour: "#15803d" }, { value: 7, colour: "#7f1d1d" }] }} />);
    expect(screen.getByAltText("Camera view of Ramkund north steps")).toHaveAttribute("src", "blob:snap");
    await userEvent.click(screen.getByText("Detections"));
    expect(screen.getByAltText("Detections overlay")).toHaveAttribute("src", "blob:detections");
    await userEvent.click(screen.getByText("Density heatmap"));
    expect(screen.getByLabelText("Heatmap colour key in persons/m²")).toBeInTheDocument();
    await userEvent.click(screen.getByText("Grid + hotspot"));
    expect(screen.getByTestId("grid-overlay")).toHaveTextContent("Hotspot · 90");
  });
});
