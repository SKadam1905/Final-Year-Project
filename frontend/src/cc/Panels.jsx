import { Suspense, lazy } from "react";
import { countMethodLabel, formatCount, formatNumber, hotspotText, prettyNode } from "../format.js";
import { SURGE_TONES, surgeStatus, surgeRuleText, turbulenceLabel, zoneState } from "../risk.js";

const CrowdTrend = lazy(() => import("./CrowdTrend.jsx")); // carries the chart library

const ACTION_ICONS = {
  normal: "✓", monitor: "👁", prepare: "👥", hold_inflow: "✋", open_exit: "↗", stop_inflow: "⛔", open_all_exits: "⇶", announce: "📢",
};

/** EXIT_GATE_C -> "Gate C"; other nodes as format.prettyNode ("Ramkund north", "Ghat 1"). */
export const placeName = (node) => (node.startsWith("EXIT_GATE_") ? `Gate ${node.slice(10)}` : prettyNode(node));

function Card({ id, title, children, aside }) {
  return (
    <section aria-labelledby={id} className="rounded-xl border border-navy-700 bg-navy-800 p-3">
      <div className="flex items-center justify-between gap-2">
        <h3 id={id} className="font-display text-sm font-semibold uppercase tracking-wider text-slate-300">{title}</h3>
        {aside}
      </div>
      <div className="mt-2">{children}</div>
    </section>
  );
}

/** Crowd estimation: the 10-minute trend, turbulence, which model counted, and the hotspot. */
export function CrowdEstimation({ zone, now, connected, thresholds, surge }) {
  if (!zone) return null;
  const live = zoneState(zone, now, connected) === "live";
  const r = zone.reading;
  const speed = live ? surgeStatus(r, surge) : null;
  return (
    <Card id="estimation-heading" title="Crowd estimation">
      <Suspense fallback={<p className="h-44 text-sm text-slate-400" aria-busy="true">Loading the chart…</p>}>
        <CrowdTrend history={zone.history} thresholds={thresholds} />
      </Suspense>
      {live ? (
        <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm">
          <dt className="text-slate-400">Turbulence</dt>
          <dd className="text-white">{formatNumber(r.turbulence)} <span className="text-slate-300">({turbulenceLabel(r.turbulence)}: how chaotic the movement is, 0–1)</span></dd>
          <dt className="text-slate-400">Counted by</dt>
          <dd className="text-white">{countMethodLabel(r.count_method)}
            <span className="text-slate-300">{r.detection_count != null ? ` · YOLO found ${formatCount(r.detection_count)}, CSRNet counted ${formatCount(r.density_count)}` : ""}</span></dd>
          <dt className="text-slate-400">Hotspot</dt>
          <dd className="text-white">{r.hotspot ? hotspotText(r.hotspot) : "none (no one counted)"}</dd>
          <dt className="text-slate-400">Speed surge</dt>
          <dd className="text-white"><span className={SURGE_TONES[speed.tone]}>{speed.text}</span>
            <span className="text-slate-300"> ({speed.note}: crowd speed vs its median over the last few seconds) {surgeRuleText(surge)}</span></dd>
        </dl>
      ) : <p className="mt-2 text-sm text-slate-400">No live reading: turbulence, counting method and hotspot are unknown.</p>}
    </Card>
  );
}

/** The safest exit for the selected zone (the backend's Dijkstra over live crowd pressure). */
export function SafestExit({ zone, now, connected }) {
  if (!zone) return null;
  const live = zoneState(zone, now, connected) === "live";
  const exits = live ? zone.reading.exits ?? [] : [];
  const actions = live ? zone.reading.gate_actions ?? [] : [];
  const best = exits[0];
  return (
    <Card id="exit-heading" title="Safest exit">
      {!live ? <p className="text-sm text-slate-400">No live reading: exit advice paused.</p>
        : !best ? <p className="text-sm text-slate-400">This camera is not on the exit map, so there is no route for it.</p> : (
          <>
            <div className="rounded-lg border-2 border-saffron bg-saffron/10 p-3">
              <p className="font-display text-3xl font-bold text-saffron">{placeName(best.exit)}</p>
              <p className="mt-1 text-sm text-slate-200">{best.path.map(placeName).join(" → ")}</p>
              <p className="text-xs text-slate-400">route cost {formatNumber(best.cost, 0)} m, weighted by live crowd pressure (schematic distances)</p>
            </div>
            {exits.length > 1 && (
              <ol className="mt-2 space-y-1 text-sm" start={2}>
                {exits.slice(1, 3).map((e, i) => (
                  <li key={e.exit} className="flex gap-2 text-slate-200">
                    <span className="font-semibold text-white">{i + 2}. {placeName(e.exit)}</span>
                    <span className="min-w-0 text-slate-400">{e.path.map(placeName).join(" → ")}</span>
                  </li>
                ))}
              </ol>
            )}
          </>
        )}
      {actions.length > 0 && (
        <div className="mt-3 border-t border-navy-600 pt-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Gate actions now</h4>
          <ul className="mt-1 space-y-1 text-sm">
            {actions.map((a) => (
              <li key={a.kind} className="flex gap-2 font-medium text-white">
                <span aria-hidden="true" className="w-5 shrink-0 text-center">{ACTION_ICONS[a.kind] ?? "•"}</span>{a.text}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}

const CAMERA = {
  online: { text: "Online", icon: "●", style: "bg-risk-green text-navy-950" },
  offline: { text: "No signal", icon: "✕", style: "bg-risk-red text-white" },
  connecting: { text: "Connecting", icon: "…", style: "bg-slate-600 text-white" },
  video: { text: "Test video", icon: "▶", style: "bg-violet-700 text-white" },
  unknown: { text: "Unknown", icon: "?", style: "bg-slate-600 text-white" },
};

export function cameraStatus(zone, health) {
  if (zone.kind === "video") return "video";
  const status = health?.cameras?.[zone.id]?.status;
  return CAMERA[status] ? status : "unknown";
}

/** Camera status: the selected camera in detail, and every camera at a glance. */
export function CameraStatus({ zone, zones, health }) {
  if (!zone) return null;
  const camera = health?.cameras?.[zone.id];
  const status = CAMERA[cameraStatus(zone, health)];
  const age = camera?.last_frame_time && health?.server_time ? health.server_time - camera.last_frame_time : null;
  return (
    <Card id="camera-heading" title="Camera status"
      aside={<span className={`rounded px-2 py-0.5 text-xs font-bold ${status.style}`}><span aria-hidden="true">{status.icon} </span>{status.text}</span>}>
      <p className="text-sm text-slate-300">
        {zone.kind === "video" ? "A recorded clip looping as a test camera, not a live feed."
          : camera ? `Last frame ${age == null ? "—" : `${formatNumber(Math.max(0, age), 1)} s`} ago · ${formatCount(camera.frames_read)} frames read · ${camera.reconnects} reconnects`
            : "No status from the camera reader yet."}
      </p>
      <ul className="mt-2 flex flex-wrap gap-1.5">
        {zones.map((z) => {
          const s = CAMERA[cameraStatus(z, health)];
          return (
            <li key={z.id} className="flex items-center gap-1 rounded bg-navy-900 px-2 py-0.5 text-xs text-slate-200">
              <span aria-hidden="true" className={`grid h-4 w-4 place-items-center rounded-full text-[9px] ${s.style}`}>{s.icon}</span>
              {z.name}<span className="sr-only">: {s.text}</span>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}
