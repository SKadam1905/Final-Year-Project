import { useState } from "react";
import { formatTime, hotspotText } from "../format.js";
import { levelUi } from "./theme.js";

/** A RED alert stays "active" (banner + repeating siren) until acknowledged, for this long at most. */
export const ACTIVE_WINDOW_S = 30 * 60;

export function activeRedAlerts(alerts, nowMs) {
  return alerts.filter((a) => a.risk_level === "RED" && !a.acknowledged_at && nowMs / 1000 - a.timestamp <= ACTIVE_WINDOW_S);
}

export function filterAlerts(alerts, { level = "ALL", unacknowledgedOnly = false }) {
  return alerts.filter((a) => (level === "ALL" || a.risk_level === level) && (!unacknowledgedOnly || !a.acknowledged_at));
}

const isTest = (alert) => alert.message?.startsWith("[VIDEO TEST]");

/** Acknowledge an alert, showing a short busy state and the backend's reason if it fails. */
function useAcknowledge(onAcknowledge) {
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  async function acknowledge(alert) {
    setBusy(alert.id);
    setError(null);
    try {
      await onAcknowledge(alert);
    } catch (e) {
      setError(`Could not acknowledge: ${e.message}`);
    } finally {
      setBusy(null);
    }
  }
  return { busy, error, acknowledge };
}

function AckButton({ alert, busy, onClick, large = false }) {
  if (alert.acknowledged_at) {
    return (
      <p className="text-xs text-green-300">
        <span aria-hidden="true">✓ </span>Acknowledged by <span className="font-semibold">{alert.acknowledged_by}</span> at {formatTime(alert.acknowledged_at * 1000)}
      </p>
    );
  }
  if (alert.id == null) return <p className="text-xs text-slate-400">Not saved, so it cannot be acknowledged.</p>;
  return (
    <button type="button" onClick={() => onClick(alert)} disabled={busy === alert.id}
      className={`rounded-md font-display font-semibold focus:outline-none focus-visible:ring-4 focus-visible:ring-white/60 disabled:opacity-60 ${
        large ? "bg-white px-4 py-2 text-base text-red-800 hover:bg-red-50" : "bg-saffron px-3 py-1 text-sm text-navy-950 hover:bg-saffron-300"}`}>
      {busy === alert.id ? "Acknowledging…" : "Acknowledge"}
    </button>
  );
}

/** Full-width banner while a RED alert is unacknowledged; the siren repeats every 30 s meanwhile. */
export function RedAlertBanner({ alerts, zoneNames, onAcknowledge, now, sound }) {
  const active = activeRedAlerts(alerts, now);
  const { busy, error, acknowledge } = useAcknowledge(onAcknowledge);
  if (!active.length) return null;
  const alert = active[0];
  return (
    <div role="alert" className="border-y-2 border-red-300 bg-red-700 text-white print:hidden">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2.5 sm:px-4">
        <p className="flex items-center gap-2 font-display text-lg font-bold sm:text-xl">
          <span aria-hidden="true" className="relative inline-flex h-4 w-4">
            <span className="absolute inset-0 animate-ping rounded-sm bg-white/70 motion-safe-only" /><span className="relative h-4 w-4 rounded-sm bg-white" />
          </span>
          {isTest(alert) && "TEST VIDEO · "}RED ALERT · {zoneNames[alert.zone_id] ?? alert.zone_id}
        </p>
        <div className="min-w-0 flex-1 basis-72 text-sm">
          <p className="font-semibold">{formatTime(alert.timestamp * 1000)}{alert.hotspot && ` · hotspot ${hotspotText(alert.hotspot)}`}</p>
          {alert.gate_actions?.[0] && <p>{alert.gate_actions[0]}</p>}
          <p className="text-xs text-red-100">
            {sound?.enabled && !sound.muted ? "The siren repeats every 30 s until acknowledged." : "Sound is off: use Enable sound in the top bar."}
            {active.length > 1 && ` ${active.length - 1} more unacknowledged RED alert${active.length > 2 ? "s" : ""} below.`}
          </p>
          {error && <p className="font-semibold text-yellow-200">{error}</p>}
        </div>
        <AckButton alert={alert} busy={busy} onClick={acknowledge} large />
      </div>
    </div>
  );
}

/** Live alert feed, newest first, filterable by level and by "unacknowledged only". */
export default function AlertFeed({ alerts, zoneNames, onAcknowledge }) {
  const [level, setLevel] = useState("ALL");
  const [unacknowledgedOnly, setUnacknowledgedOnly] = useState(false);
  const { busy, error, acknowledge } = useAcknowledge(onAcknowledge);
  const shown = filterAlerts(alerts, { level, unacknowledgedOnly });
  const waiting = alerts.filter((a) => !a.acknowledged_at).length;

  return (
    <section aria-labelledby="alerts-heading" className="rounded-xl border border-navy-700 bg-navy-800 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="alerts-heading" className="font-display text-sm font-semibold uppercase tracking-wider text-slate-300">
          Alerts <span className="font-normal normal-case text-slate-400">({alerts.length}, {waiting} unacknowledged)</span>
        </h2>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <div role="group" aria-label="Filter by level" className="flex overflow-hidden rounded-md ring-1 ring-navy-500">
            {["ALL", "RED", "ORANGE"].map((value) => (
              <button key={value} type="button" aria-pressed={level === value} onClick={() => setLevel(value)}
                className={`px-2.5 py-1 font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-saffron ${
                  level === value ? "bg-saffron text-navy-950" : "bg-navy-900 text-slate-300 hover:text-white"}`}>
                {value === "ALL" ? "All" : `${levelUi(value).icon} ${value}`}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-1.5 text-slate-200">
            <input type="checkbox" checked={unacknowledgedOnly} onChange={(e) => setUnacknowledgedOnly(e.target.checked)} className="h-4 w-4 accent-saffron" />
            Unacknowledged only
          </label>
        </div>
      </div>
      {error && <p role="alert" className="mt-2 rounded border-l-4 border-risk-red bg-risk-red/15 px-2 py-1 text-sm text-red-100">{error}</p>}
      {shown.length === 0 ? (
        <p className="mt-3 text-sm text-slate-400">{alerts.length ? "No alerts match these filters." : "No alerts yet: they appear here as soon as a zone reaches ORANGE or RED."}</p>
      ) : (
        <ol className="mt-2 max-h-[22rem] space-y-2 overflow-y-auto pr-1">
          {shown.map((alert) => {
            const ui = levelUi(alert.risk_level);
            const failed = alert.deliveries?.some((d) => !d.delivered);
            return (
              <li key={alert.id ?? `${alert.zone_id}@${alert.timestamp}`} className="animate-fade-in rounded-lg border-l-4 bg-navy-900 p-2.5"
                style={{ borderLeftColor: ui.fill }}>
                <div className="flex flex-wrap items-center gap-2">
                  <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-display text-xs font-bold ${ui.chip}`}>
                    <span aria-hidden="true">{ui.icon}</span>{alert.risk_level}
                  </span>
                  <span className="font-semibold text-white">{zoneNames[alert.zone_id] ?? alert.zone_id}</span>
                  {isTest(alert) && <span className="rounded bg-violet-700 px-1 text-[10px] font-bold uppercase text-white">test video</span>}
                  {alert.reason && <span className="rounded bg-orange-800 px-1 text-[10px] font-bold uppercase text-white">⚡ {alert.reason}</span>}
                  <time className="ml-auto text-xs tabular-nums text-slate-400">{formatTime(alert.timestamp * 1000)}</time>
                </div>
                {alert.hotspot && <p className="mt-1 text-xs text-slate-300"><span className="font-semibold text-saffron">Hotspot:</span> {hotspotText(alert.hotspot)}</p>}
                {alert.gate_actions?.length > 0 && (
                  <ul className="mt-1 list-inside list-disc text-xs text-slate-200">
                    {alert.gate_actions.map((action) => <li key={action}>{action}</li>)}
                  </ul>
                )}
                {failed && <p className="mt-1 text-xs text-yellow-200">⚠ A delivery failed (e.g. Telegram unreachable).</p>}
                <div className="mt-1.5"><AckButton alert={alert} busy={busy} onClick={acknowledge} /></div>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
