import { formatTime } from "../format.js";
import { alertKey } from "../liveState.js";
import { riskStyle } from "../risk.js";
import RiskBadge from "./RiskBadge.jsx";

function failedChannels(alert) {
  return alert.deliveries.filter((d) => !d.delivered).map((d) => `${d.channel}${d.error ? ` (${d.error})` : ""}`);
}

/** Recent alerts, newest first. */
export default function AlertPanel({ alerts, zoneNames }) {
  return (
    <aside aria-labelledby="alerts-heading" className="rounded-xl bg-white p-4 shadow-sm">
      <h2 id="alerts-heading" className="text-lg font-bold">Recent alerts</h2>
      {alerts.length === 0 ? (
        <p className="mt-2 text-sm text-slate-700">No alerts yet.</p>
      ) : (
        <ol className="mt-3 space-y-3">
          {alerts.map((alert) => {
            const failed = failedChannels(alert);
            return (
              <li key={alertKey(alert)} className="border-t border-slate-200 pt-3 first:border-t-0 first:pt-0">
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <RiskBadge level={riskStyle(alert.risk_level)} size="sm" />
                  <time className="font-semibold tabular-nums">{formatTime(alert.timestamp * 1000)}</time>
                  <span className="font-semibold">{zoneNames[alert.zone_id] ?? alert.zone_id}</span>
                </div>
                <p className="mt-1 whitespace-pre-line text-sm text-slate-900">{alert.message}</p>
                {failed.length > 0 && (
                  <p className="mt-1 text-sm font-semibold text-red-800">Not delivered: {failed.join(", ")}</p>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </aside>
  );
}
