// Audible alerts, synthesised with the Web Audio API (no sound files): a rising-and-falling siren
// for RED, a short double beep for ORANGE. Browsers only allow sound after the user has clicked
// something, hence the "Enable sound" button in the header.

import { useCallback, useEffect, useRef, useState } from "react";
import { alertKey } from "./liveState.js";

const MUTE_KEY = "kumbh.soundMuted";

/**
 * Which sound the newest alerts call for. `alerts` are newest first; `played` is the Set of alert
 * keys already handled. Returns the level to play ("RED" beats "ORANGE", null for none) and the
 * keys of the alerts this covers, so the same alert never sounds twice.
 */
export function soundFor(alerts, played) {
  const fresh = alerts.filter((a) => !played.has(alertKey(a)));
  const levels = fresh.map((a) => a.risk_level);
  const level = levels.includes("RED") ? "RED" : levels.includes("ORANGE") ? "ORANGE" : null;
  return { level, keys: fresh.map(alertKey) };
}

function beep(ctx, start, duration, frequency, type = "sine", volume = 0.25) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(frequency, start);
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(volume, start + 0.02);
  gain.gain.setValueAtTime(volume, start + duration - 0.03);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
  osc.connect(gain).connect(ctx.destination);
  osc.start(start);
  osc.stop(start + duration);
  return osc;
}

/** RED: about 3 s of a siren sweeping 650 -> 1300 -> 650 Hz, three times. */
export function playSiren(ctx) {
  const t0 = ctx.currentTime + 0.05;
  const osc = beep(ctx, t0, 3.0, 650, "sawtooth", 0.18);
  for (let i = 0; i < 3; i += 1) {
    osc.frequency.linearRampToValueAtTime(1300, t0 + i + 0.5);
    osc.frequency.linearRampToValueAtTime(650, t0 + i + 1.0);
  }
}

/** ORANGE: two short 880 Hz beeps (about 0.5 s). */
export function playTone(ctx) {
  const t0 = ctx.currentTime + 0.05;
  beep(ctx, t0, 0.18, 880);
  beep(ctx, t0 + 0.3, 0.18, 880);
}

function readMuted() {
  try {
    return window.localStorage.getItem(MUTE_KEY) === "1";
  } catch {
    return false;
  }
}

function saveMuted(muted) {
  try {
    window.localStorage.setItem(MUTE_KEY, muted ? "1" : "0");
  } catch {
    // private window or blocked storage: the setting lasts for this page only
  }
}

/**
 * Plays each new ORANGE/RED alert once. Until sound is enabled (a click), and while muted, new
 * alerts are marked as handled silently, so enabling or unmuting never replays old ones.
 */
export function useAlertSound(alerts, AudioContextClass = globalThis.AudioContext ?? globalThis.webkitAudioContext) {
  const [enabled, setEnabled] = useState(false);
  const [muted, setMuted] = useState(readMuted);
  const ctx = useRef(null);
  const played = useRef(new Set());
  const supported = Boolean(AudioContextClass);

  useEffect(() => {
    const { level, keys } = soundFor(alerts, played.current);
    keys.forEach((key) => played.current.add(key));
    if (!level || !enabled || muted || !ctx.current) return;
    if (level === "RED") playSiren(ctx.current);
    else playTone(ctx.current);
  }, [alerts, enabled, muted]);

  const enable = useCallback(() => {
    if (!supported) return;
    ctx.current = ctx.current ?? new AudioContextClass();
    ctx.current.resume?.();
    setEnabled(true);
    playTone(ctx.current); // confirms that sound works
  }, [AudioContextClass, supported]);

  /** Play a level's sound now (used to repeat the RED siren); silent until enabled, and while muted. */
  const play = useCallback((level) => {
    if (!enabled || muted || !ctx.current) return;
    if (level === "RED") playSiren(ctx.current);
    else playTone(ctx.current);
  }, [enabled, muted]);

  const toggleMute = useCallback(() => {
    setMuted((m) => {
      saveMuted(!m);
      return !m;
    });
  }, []);

  return { supported, enabled, muted, enable, toggleMute, play };
}

/** While `active` (an unacknowledged RED alert), sound the siren again every `everyMs`. */
export function useRepeatingSiren(active, sound, everyMs = 30_000) {
  const play = sound?.play;
  useEffect(() => {
    if (!active || !play) return undefined;
    const timer = setInterval(() => play("RED"), everyMs);
    return () => clearInterval(timer);
  }, [active, play, everyMs]);
}
