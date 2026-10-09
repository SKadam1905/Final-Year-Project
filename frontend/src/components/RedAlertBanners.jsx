import { formatTime } from "../format.js";
import { alertKey } from "../liveState.js";

/** Full-width banner for each live RED alert, until a guard dismisses it. */
export default function RedAlertBanners({ alerts, zoneNames, onDismiss }) {
  if (alerts.length === 0) return null;
  return (
    <div className="space-y-px">
      {alerts.map((alert) => (
        <div key={alertKey(alert)} role="alert" className="bg-red-700 text-white">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-2 px-3 py-3 sm:px-6">
            <p className="text-xl font-extrabold">
              <span aria-hidden="true">■ </span>
              {alert.message?.startsWith("[VIDEO TEST]") && "VIDEO TEST, not a live incident · "}
              RED ALERT · {zoneNames[alert.zone_id] ?? alert.zone_id}
            </p>
            <p className="min-w-0 flex-1 basis-64 whitespace-pre-line text-sm font-semibold">
              <time className="tabular-nums">{formatTime(alert.timestamp * 1000)}</time> — {alert.message}
            </p>
            <button
              type="button"
              onClick={() => onDismiss(alertKey(alert))}
              className="rounded-md bg-white px-4 py-2 font-bold text-red-800 focus:outline-none focus-visible:ring-4 focus-visible:ring-yellow-300"
            >
              Dismiss
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
