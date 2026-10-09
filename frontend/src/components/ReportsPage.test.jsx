import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import ReportsPage from "./ReportsPage.jsx";

const ZONES = [{ id: "GHAT_1", name: "Ghat 1 approach" }, { id: "RAMKUND_NORTH", name: "Ramkund north steps" }];
const SUMMARY = {
  storage: "mongodb", readings: 120, zones: [{
    zone_id: "GHAT_1", name: "Ghat 1 approach", readings: 120, first: 1_800_000_000, last: 1_800_000_240,
    peak_count: 812.4, peak_count_time: 1_800_000_100, peak_density: 6.4, peak_density_time: 1_800_000_100,
    peak_pressure: 6.9, highest_level: "RED", time_at_level_s: { GREEN: 100, YELLOW: 60, ORANGE: 50, RED: 30 },
    detection_readings: 0,
  }],
};
const ALERTS = [{ id: "a1", zone_id: "GHAT_1", risk_level: "RED", message: "RED: Ghat 1 approach\nGate actions:\n- STOP inflow",
  timestamp: 1_800_000_100, delivered: true }];
const INCIDENTS = [{ id: "i1", zone_id: "GHAT_1", started_at: "2027-01-15T03:30:00+00:00", ended_at: null,
  peak_level: "RED", peak_pressure: 6.9, peak_density: 6.4, peak_count: 812.4 }];

function fakeApi() {
  return {
    summary: vi.fn().mockResolvedValue(SUMMARY),
    alerts: vi.fn().mockResolvedValue(ALERTS),
    incidents: vi.fn().mockResolvedValue(INCIDENTS),
  };
}

describe("ReportsPage", () => {
  it("loads the last hour of ORANGE/RED history and shows summary, alerts and incidents", async () => {
    const api = fakeApi();
    render(<ReportsPage zones={ZONES} api={api} />);
    expect(await screen.findByRole("heading", { name: /Per-zone summary/ })).toBeInTheDocument();
    expect(api.alerts.mock.calls[0][0]).toMatch(/^\?level=ORANGE&level=RED&since=\d+&until=\d+$/);
    expect(api.summary.mock.calls[0][0]).not.toContain("level=");

    const summary = screen.getByRole("heading", { name: /Per-zone summary/ }).closest("section");
    expect(within(summary).getByText("812")).toBeInTheDocument();
    expect(within(summary).getByText("1 min 40 s")).toBeInTheDocument(); // time GREEN
    expect(screen.getByText(/STOP inflow/)).toBeInTheDocument();
    expect(screen.getByText("ongoing")).toBeInTheDocument();
  });

  it("filters by zone and level when Show report is pressed", async () => {
    const api = fakeApi();
    render(<ReportsPage zones={ZONES} api={api} />);
    await screen.findByRole("heading", { name: /Alert history/ });
    await userEvent.selectOptions(screen.getByLabelText("Zone"), "GHAT_1");
    await userEvent.click(screen.getByLabelText("ORANGE"));
    await userEvent.click(screen.getByRole("button", { name: "Show report" }));
    expect(api.alerts).toHaveBeenCalledTimes(2);
    expect(api.alerts.mock.calls[1][0]).toMatch(/^\?zone_id=GHAT_1&level=RED&since=/);
  });

  it("shows why a report is unavailable", async () => {
    const api = { ...fakeApi(), summary: vi.fn().mockRejectedValue(new Error("history unavailable: timed out")) };
    render(<ReportsPage zones={ZONES} api={api} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Report unavailable: history unavailable: timed out");
  });
});
