import { useEffect, useState } from "react";
import { API_URL } from "../api.js";
import { formatTime } from "../format.js";
import { sortZones } from "../risk.js";
import AlertFeed from "./AlertFeed.jsx";
import GhatMap from "./GhatMap.jsx";
import KpiStrip from "./KpiStrip.jsx";
import { computeKpis } from "./kpi.js";
import ZoneList from "./ZoneList.jsx";
import LiveMonitor from "./LiveMonitor.jsx";
import { CameraStatus, CrowdEstimation, SafestExit } from "./Panels.jsx";
import ZoneSummary from "./ZoneSummary.jsx";

/** Why the screen may be incomplete: backend down, socket disconnected, no model. Never silent. */
function Notices({ backend, connection, lastUpdate }) {
  const items = [];
  if (backend.status === "unreachable") {
    items.push(["red", `Cannot reach the backend at ${API_URL}: retrying every 5 s. ${backend.error ?? ""}`]);
  } else if (connection === "disconnected") {
    items.push(["red", `Disconnected from the live feed: the figures below are frozen at ${formatTime(lastUpdate)} and greyed out. Reconnecting…`]);
  }
  if (backend.health && backend.health.model_loaded === false) {
    items.push(["orange", `No trained model: counts unavailable. ${backend.health.model_error ?? ""}`]);
  }
  if (!items.length) return null;
  return (
    <div className="space-y-1 px-3 pt-2 sm:px-4">
      {items.map(([tone, text]) => (
        <p key={text} role="alert" className={`rounded-lg border-l-4 px-3 py-2 text-sm font-medium ${
          tone === "red" ? "border-risk-red bg-risk-red/15 text-red-100" : "border-risk-orange bg-risk-orange/15 text-orange-100"}`}>
          {text}
        </p>
      ))}
    </div>
  );
}

/** The Command Centre: live KPIs, the zones, and the selected zone's details. */
export default function CommandCentre({ live, selectedId, onSelect }) {
  const { state, now, connection, backend, loadHistory, acknowledge } = live;
  const connected = connection === "live";
  const zones = sortZones(Object.values(state.zones), now, connected);
  const kpis = computeKpis({ zones, alerts: state.alerts, health: backend.health, now, connected });
  const [firstPick, setFirstPick] = useState(true);
  const selected = state.zones[selectedId];

  useEffect(() => {
    if (selectedId) loadHistory(selectedId); // the last 10 minutes for the trend chart
  }, [selectedId, loadHistory]);

  useEffect(() => {
    // Start on the highest-risk zone once the first data arrives; afterwards the user decides.
    if (firstPick && zones.length && zones[0].reading) {
      if (!selectedId) onSelect(zones[0].id);
      setFirstPick(false);
    }
  }, [firstPick, zones, selectedId, onSelect]);

  return (
    <div className="pb-6">
      <Notices backend={backend} connection={connection} lastUpdate={state.lastUpdate} />
      <div className={`transition ${connected ? "" : "opacity-60 grayscale"}`} aria-busy={connection === "connecting"}>
        <div className="pt-3"><KpiStrip kpis={kpis} health={backend.health} /></div>
        <main className="grid gap-3 px-3 pt-3 sm:px-4 lg:grid-cols-[minmax(0,1fr)_minmax(340px,410px)]">
          <div className="min-w-0 space-y-3">
            <GhatMap zones={zones} now={now} connected={connected} thresholds={backend.config?.thresholds}
              selectedId={selectedId} onSelect={onSelect} />
            <AlertFeed alerts={state.alerts} zoneNames={Object.fromEntries(zones.map((z) => [z.id, z.name]))} onAcknowledge={acknowledge} />
            <ZoneList zones={zones} now={now} connected={connected} selectedId={selectedId} onSelect={onSelect} />
          </div>
          <aside aria-label="Selected zone"
            className="min-w-0 space-y-3 lg:sticky lg:top-[4.5rem] lg:max-h-[calc(100vh-5rem)] lg:self-start lg:overflow-y-auto lg:pr-1">
            <ZoneSummary zone={selected} now={now} connected={connected} />
            <LiveMonitor key={selectedId} zone={selected} densityScale={backend.config?.density_scale} />
            <CrowdEstimation zone={selected} now={now} connected={connected} thresholds={backend.config?.thresholds} />
            <SafestExit zone={selected} now={now} connected={connected} />
            <CameraStatus zone={selected} zones={zones} health={backend.health} />
          </aside>
        </main>
      </div>
    </div>
  );
}
