import { act, render, renderHook, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ackAlert, initialState } from "../liveState.js";
import { useRepeatingSiren } from "../sound.js";
import AlertFeed, { RedAlertBanner, activeRedAlerts, filterAlerts } from "./AlertFeed.jsx";

const NOW = 1_800_000_000_000;
const alert = (id, level, ageS = 30, extra = {}) => ({
  id, zone_id: "GHAT_1", risk_level: level, message: `${level}: Ghat 1 approach`, timestamp: NOW / 1000 - ageS,
  delivered: true, deliveries: [], hotspot: { label: "top-right", count: 90, density: 6.1 },
  gate_actions: ["STOP inflow at all gates feeding this zone: Gate C, Gate A.", "Open all exits: Gate A, Gate B, Gate C, Gate D."],
  acknowledged_by: null, acknowledged_at: null, ...extra,
});
const NAMES = { GHAT_1: "Ghat 1 approach" };

afterEach(() => vi.useRealTimers());

describe("alert helpers", () => {
  it("filters by level and by unacknowledged", () => {
    const list = [alert(1, "RED"), alert(2, "ORANGE"), alert(3, "RED", 30, { acknowledged_at: 5, acknowledged_by: "officer" })];
    expect(filterAlerts(list, { level: "RED" }).map((a) => a.id)).toEqual([1, 3]);
    expect(filterAlerts(list, { unacknowledgedOnly: true }).map((a) => a.id)).toEqual([1, 2]);
  });

  it("keeps a RED alert active until acknowledged, for 30 minutes at most", () => {
    const list = [alert(1, "RED", 60), alert(2, "RED", 31 * 60), alert(3, "ORANGE"), alert(4, "RED", 10, { acknowledged_at: 1 })];
    expect(activeRedAlerts(list, NOW).map((a) => a.id)).toEqual([1]);
  });

  it("merges an acknowledgement into the alert list", () => {
    const state = { ...initialState, alerts: [alert(1, "RED"), alert(2, "ORANGE")] };
    const next = ackAlert(state, { ...alert(1, "RED"), acknowledged_by: "officer", acknowledged_at: 123 });
    expect(next.alerts[0]).toMatchObject({ id: 1, acknowledged_by: "officer", acknowledged_at: 123 });
    expect(next.alerts[1].acknowledged_at).toBeNull();
  });
});

describe("AlertFeed", () => {
  it("lists level, place, time, hotspot and gate actions, and acknowledges", async () => {
    const onAcknowledge = vi.fn().mockResolvedValue(undefined);
    render(<AlertFeed alerts={[alert(1, "RED"), alert(2, "ORANGE", 90, { acknowledged_by: "admin", acknowledged_at: NOW / 1000 })]}
      zoneNames={NAMES} onAcknowledge={onAcknowledge} />);
    expect(screen.getByText(/2, 1 unacknowledged/)).toBeInTheDocument();
    expect(screen.getAllByText(/top-right of the view/)).toHaveLength(2);
    expect(screen.getAllByText("STOP inflow at all gates feeding this zone: Gate C, Gate A.")).toHaveLength(2);
    expect(screen.getByText(/Acknowledged by/)).toHaveTextContent("Acknowledged by admin at");
    await userEvent.click(screen.getByRole("button", { name: "Acknowledge" }));
    expect(onAcknowledge).toHaveBeenCalledWith(expect.objectContaining({ id: 1 }));
  });

  it("filters, and says so when nothing matches", async () => {
    render(<AlertFeed alerts={[alert(2, "ORANGE")]} zoneNames={NAMES} onAcknowledge={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /RED/ }));
    expect(screen.getByText("No alerts match these filters.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "All" }));
    await userEvent.click(screen.getByLabelText("Unacknowledged only"));
    expect(screen.getByText("Ghat 1 approach")).toBeInTheDocument();
  });

  it("shows why an acknowledgement failed", async () => {
    render(<AlertFeed alerts={[alert(1, "RED")]} zoneNames={NAMES} onAcknowledge={vi.fn().mockRejectedValue(new Error("no alert '1'"))} />);
    await userEvent.click(screen.getByRole("button", { name: "Acknowledge" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not acknowledge: no alert '1'");
  });
});

describe("RedAlertBanner", () => {
  it("shows the newest unacknowledged RED alert with Acknowledge, and nothing otherwise", async () => {
    const onAcknowledge = vi.fn().mockResolvedValue(undefined);
    const { rerender } = render(<RedAlertBanner alerts={[alert(1, "RED"), alert(2, "RED", 120)]} zoneNames={NAMES}
      onAcknowledge={onAcknowledge} now={NOW} sound={{ enabled: true, muted: false }} />);
    expect(screen.getByRole("alert")).toHaveTextContent("RED ALERT · Ghat 1 approach");
    expect(screen.getByRole("alert")).toHaveTextContent("The siren repeats every 30 s until acknowledged. 1 more unacknowledged RED alert below.");
    await userEvent.click(screen.getByRole("button", { name: "Acknowledge" }));
    expect(onAcknowledge).toHaveBeenCalledWith(expect.objectContaining({ id: 1 }));
    rerender(<RedAlertBanner alerts={[alert(3, "ORANGE")]} zoneNames={NAMES} onAcknowledge={onAcknowledge} now={NOW} sound={null} />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("useRepeatingSiren", () => {
  it("repeats the siren every 30 s while active, and stops once acknowledged", () => {
    vi.useFakeTimers();
    const play = vi.fn();
    const { rerender } = renderHook(({ active }) => useRepeatingSiren(active, { play }), { initialProps: { active: true } });
    act(() => vi.advanceTimersByTime(65_000));
    expect(play).toHaveBeenCalledTimes(2);
    expect(play).toHaveBeenCalledWith("RED");
    rerender({ active: false });
    act(() => vi.advanceTimersByTime(120_000));
    expect(play).toHaveBeenCalledTimes(2);
  });
});
