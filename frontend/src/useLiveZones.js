// All network I/O for the dashboard lives here: the Socket.IO connection, REST bootstrap and
// the per-zone image feed. Components receive plain data and callbacks.

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { io } from "socket.io-client";
import { API_URL, api, fetchImage, imageUrl, socketOptions, socketRefused } from "./api.js";
import { initialState, reducer } from "./liveState.js";

const RETRY_MS = 5000;
const HEALTH_MS = 5000;

/**
 * Live dashboard state. `connection` is "connecting" | "live" | "disconnected";
 * `backend.status` is "loading" | "ok" | "unreachable"; `now` ticks every second so the
 * 10-second stale rule is re-evaluated even when no events arrive.
 */
export function useLiveZones() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [connection, setConnection] = useState("connecting");
  const [backend, setBackend] = useState({ status: "loading", health: null, config: null, error: null });
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  const load = useCallback(async () => {
    try {
      const [health, config, zones, alerts] = await Promise.all([
        api.health(), api.config(), api.zones(), api.recentAlerts(),
      ]);
      dispatch({ type: "zones", zones });
      dispatch({ type: "alerts", alerts, serverTime: health.server_time, bannerWindowS: config.alert_cooldown_s });
      setBackend({ status: "ok", health, config, error: null });
    } catch (error) {
      setBackend((previous) => ({ ...previous, status: "unreachable", error: error.message }));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Camera status, model and cycle time change without a socket event: refresh /health every 5 s.
  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        const health = await api.health();
        setBackend((previous) => (previous.status === "ok" ? { ...previous, health } : previous));
      } catch {
        // the socket and the retry below report an unreachable backend
      }
    }, HEALTH_MS);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (backend.status !== "unreachable") return undefined;
    const timer = setTimeout(load, RETRY_MS);
    return () => clearTimeout(timer);
  }, [backend, load]);

  useEffect(() => {
    // Connect one tick later: React's development double-mount then never opens a socket it closes
    // at once (which the browser reports as "WebSocket is closed before the connection is established").
    const socket = io(API_URL, socketOptions({ reconnectionDelayMax: RETRY_MS }));
    const connectTimer = setTimeout(() => socket.connect(), 0);
    socket.on("connect", () => {
      setConnection("live");
      load(); // the backend may have restarted with different zones or a model
    });
    socket.on("disconnect", () => setConnection("disconnected"));
    socket.on("connect_error", (error) => {
      setConnection("disconnected");
      if (socketRefused(error)) socket.disconnect();
    });
    socket.on("density_update", (payload) => dispatch({ type: "density", payload, now: Date.now() }));
    socket.on("alert", (alert) => dispatch({ type: "alert", alert }));
    socket.on("zones_changed", (change) => {
      if (change?.removed) dispatch({ type: "removeZone", zoneId: change.removed });
      load(); // an uploaded video was added or removed as a live zone
    });
    return () => {
      clearTimeout(connectTimer);
      socket.disconnect();
    };
  }, [load]);

  const dismissBanner = useCallback((key) => dispatch({ type: "dismiss", key }), []);

  const loadHistory = useCallback(async (zoneId) => {
    try {
      const body = await api.history(zoneId, 300); // 10 minutes at one reading every 2 s
      dispatch({ type: "history", zoneId, readings: body.readings });
    } catch {
      // The chart still fills from live updates.
    }
  }, []);

  return { state, now, connection, backend, dismissBanner, loadHistory };
}

const EMPTY_FEED = { snapshot: null, overlay: null, snapshotError: null, overlayError: null, capturedAt: null };

/**
 * Camera snapshot for one zone plus an optional overlay ("none" | "density" | "gradcam"),
 * refreshed every `refreshMs`. Images are object URLs; old ones are revoked once replaced.
 */
export function useZoneFeed(zoneId, overlay, refreshMs = 2000) {
  const [feed, setFeed] = useState(EMPTY_FEED);
  const urls = useRef([]);

  useEffect(() => {
    if (!zoneId) return undefined;
    let cancelled = false;
    setFeed(EMPTY_FEED);

    const grab = (kind) =>
      fetchImage(imageUrl(zoneId, kind)).then(
        (image) => ({ image }),
        (error) => ({ error: error.message }),
      );

    async function refresh() {
      const [snap, over] = await Promise.all([grab("snapshot"), overlay === "none" ? null : grab(overlay)]);
      const created = [snap.image?.url, over?.image?.url].filter(Boolean);
      if (cancelled) {
        created.forEach((url) => URL.revokeObjectURL(url));
        return;
      }
      urls.current.push(...created);
      while (urls.current.length > 4) URL.revokeObjectURL(urls.current.shift()); // keep the last two pairs
      setFeed((previous) => ({
        snapshot: snap.image?.url ?? null,
        snapshotError: snap.error ?? null,
        capturedAt: snap.image?.timestamp ?? previous.capturedAt,
        overlay: over?.image?.url ?? null,
        overlayError: over?.error ?? null,
      }));
    }

    refresh();
    const timer = setInterval(refresh, refreshMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
      // Revoke after React has swapped the images out; revoking now breaks the visible <img>.
      const stale = urls.current;
      urls.current = [];
      setTimeout(() => stale.forEach((url) => URL.revokeObjectURL(url)), refreshMs);
    };
  }, [zoneId, overlay, refreshMs]);

  return feed;
}
