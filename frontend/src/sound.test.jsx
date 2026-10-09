import { act, render, renderHook, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import Header from "./components/Header.jsx";
import { soundFor, useAlertSound } from "./sound.js";

/** Records the oscillators started: their type tells the siren (sawtooth) from the tone (sine). */
class FakeAudioContext {
  static started = [];

  currentTime = 0;
  destination = {};

  resume() {}

  createGain() {
    const param = { setValueAtTime() {}, exponentialRampToValueAtTime() {} };
    return { gain: param, connect: (node) => node };
  }

  createOscillator() {
    const frequency = { setValueAtTime() {}, linearRampToValueAtTime() {} };
    const osc = { type: "sine", frequency, connect: (node) => node, stop() {} };
    osc.start = () => FakeAudioContext.started.push(osc.type);
    return osc;
  }
}

const alert = (id, risk_level) => ({ id, zone_id: "GHAT_1", risk_level, message: "", timestamp: id });

beforeEach(() => {
  FakeAudioContext.started = [];
  window.localStorage.clear();
});

describe("soundFor", () => {
  it("picks the most severe new alert and skips ones already played", () => {
    expect(soundFor([alert(2, "ORANGE"), alert(1, "RED")], new Set())).toEqual({ level: "RED", keys: [2, 1] });
    expect(soundFor([alert(2, "ORANGE"), alert(1, "RED")], new Set([1]))).toEqual({ level: "ORANGE", keys: [2] });
    expect(soundFor([alert(1, "RED")], new Set([1]))).toEqual({ level: null, keys: [] });
  });
});

describe("useAlertSound", () => {
  it("stays silent until enabled, then plays each new alert once (siren for RED, tone for ORANGE)", () => {
    const { result, rerender } = renderHook(({ alerts }) => useAlertSound(alerts, FakeAudioContext),
      { initialProps: { alerts: [alert(1, "RED")] } });
    expect(FakeAudioContext.started).toEqual([]); // alerts from before the click never sound

    act(() => result.current.enable());
    expect(FakeAudioContext.started).toEqual(["sine", "sine"]); // the confirmation tone
    expect(result.current.enabled).toBe(true);

    FakeAudioContext.started = [];
    rerender({ alerts: [alert(2, "RED"), alert(1, "RED")] });
    expect(FakeAudioContext.started).toEqual(["sawtooth"]);
    rerender({ alerts: [alert(2, "RED"), alert(1, "RED")] }); // same alerts again: no replay
    rerender({ alerts: [alert(3, "ORANGE"), alert(2, "RED"), alert(1, "RED")] });
    expect(FakeAudioContext.started).toEqual(["sawtooth", "sine", "sine"]);
  });

  it("mutes, remembers it, and does not replay what arrived while muted", () => {
    const { result, rerender } = renderHook(({ alerts }) => useAlertSound(alerts, FakeAudioContext),
      { initialProps: { alerts: [] } });
    act(() => result.current.enable());
    act(() => result.current.toggleMute());
    FakeAudioContext.started = [];
    rerender({ alerts: [alert(1, "RED")] });
    expect(FakeAudioContext.started).toEqual([]);
    expect(window.localStorage.getItem("kumbh.soundMuted")).toBe("1");
    act(() => result.current.toggleMute());
    expect(FakeAudioContext.started).toEqual([]); // unmuting does not play the missed alert
  });
});

describe("Header sound control", () => {
  function HeaderWithSound({ alerts }) {
    const sound = useAlertSound(alerts, FakeAudioContext);
    return <Header connection="live" lastUpdate={null} sound={sound} />;
  }

  it("offers Enable sound, then a mute toggle", async () => {
    render(<HeaderWithSound alerts={[]} />);
    await userEvent.click(screen.getByRole("button", { name: /Enable sound/ }));
    const toggle = screen.getByRole("button", { name: /Sound on/ });
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    await userEvent.click(toggle);
    expect(screen.getByRole("button", { name: /Sound muted/ })).toHaveAttribute("aria-pressed", "true");
  });
});
