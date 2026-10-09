import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useReducer } from "react";
import { describe, expect, it } from "vitest";
import { activeBanners, initialState, reducer } from "../liveState.js";
import AlertPanel from "./AlertPanel.jsx";
import RedAlertBanners from "./RedAlertBanners.jsx";

const NAMES = { Z1: "Ramkund north steps", Z2: "Ghat 1 approach" };

function alert(id, level, timestamp, zoneId = "Z1", deliveries = [{ channel: "console", delivered: true }]) {
  return { id, zone_id: zoneId, risk_level: level, message: `${level}: crowd building`, timestamp,
    delivered: deliveries.some((d) => d.delivered), deliveries };
}

/** The alert panel and banners on the real reducer; returns a function that fires a live alert. */
function renderAlerts(restAlerts = []) {
  let dispatch;
  function Harness() {
    const [state, send] = useReducer(reducer, reducer(initialState, { type: "alerts", alerts: restAlerts }));
    dispatch = send;
    return (
      <>
        <RedAlertBanners alerts={activeBanners(state)} zoneNames={NAMES} onDismiss={(key) => send({ type: "dismiss", key })} />
        <AlertPanel alerts={state.alerts} zoneNames={NAMES} />
      </>
    );
  }
  render(<Harness />);
  return (live) => act(() => dispatch({ type: "alert", alert: live }));
}

const listItems = () => within(screen.getByRole("complementary")).getAllByRole("listitem");

describe("alert panel and RED banner", () => {
  it("lists REST alerts and prepends live ones", () => {
    const fire = renderAlerts([alert(1, "YELLOW", 100)]);
    expect(listItems()).toHaveLength(1);

    fire(alert(2, "ORANGE", 200, "Z2"));
    const items = listItems();
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent("ORANGE");
    expect(items[0]).toHaveTextContent("Ghat 1 approach");
    expect(items[1]).toHaveTextContent("YELLOW");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument(); // no banner below RED
  });

  it("shows a RED banner that stays until dismissed", async () => {
    const fire = renderAlerts();
    fire(alert(3, "RED", 300));

    const banner = screen.getByRole("alert");
    expect(banner).toHaveTextContent("RED ALERT · Ramkund north steps");
    expect(banner).toHaveTextContent("RED: crowd building");
    fire(alert(4, "YELLOW", 310)); // later, lower alerts do not clear it
    expect(screen.getByRole("alert")).toBeInTheDocument();

    await userEvent.click(within(banner).getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(listItems()).toHaveLength(2); // dismissed banners stay in the list
  });

  it("flags alerts that a channel failed to deliver", () => {
    const failed = alert(5, "ORANGE", 400, "Z1", [{ channel: "console", delivered: true },
      { channel: "telegram", delivered: false, error: "chat not found" }]);
    render(<AlertPanel alerts={[failed]} zoneNames={NAMES} />);
    expect(screen.getByText("Not delivered: telegram (chat not found)")).toBeInTheDocument();
  });
});
