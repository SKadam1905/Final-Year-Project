import { countMethodLabel, formatCount, formatNumber, formatTime, hotspotText, prettyNode } from "../format.js";
import { displayLevel, turbulenceLabel, zoneState } from "../risk.js";
import RiskBadge from "./RiskBadge.jsx";

function Metric({ label, value }) {
  return (
    <div>
      <dt className="text-xs font-semibold uppercase tracking-wide text-slate-600">{label}</dt>
      <dd className="text-lg font-bold tabular-nums text-slate-900">{value}</dd>
    </div>
  );
}

function Exits({ exits }) {
  if (!exits?.length) return <p className="mt-3 text-sm text-slate-700">No exit route for this camera.</p>;
  return (
    <div className="mt-3">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-600">Recommended exits</h4>
      <ol className="mt-1 space-y-1 text-sm">
        {exits.map((e, i) => (
          <li key={e.exit} className="text-slate-900">
            <span className="font-bold">{i + 1}. {prettyNode(e.exit)}</span>
            <span className="text-slate-700"> — {e.path.map(prettyNode).join(" → ")}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

const ACTION_ICONS = {
  normal: "✓", monitor: "👁", prepare: "👥", hold_inflow: "✋", open_exit: "↗", stop_inflow: "⛔",
  open_all_exits: "⇶", announce: "📢",
};

/** What gate staff should do now, from the backend (app/gate_actions.py). */
function GateActions({ actions, level }) {
  if (!actions?.length) return null;
  return (
    <div className={`mt-3 rounded-lg border-l-8 bg-slate-50 p-2 ${level.border}`}>
      <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-600">Gate actions</h4>
      <ul className="mt-1 space-y-1 text-sm">
        {actions.map((a) => (
          <li key={a.kind} className="flex gap-2 font-semibold text-slate-900">
            <span aria-hidden="true" className="w-5 shrink-0 text-center">{ACTION_ICONS[a.kind] ?? "•"}</span>
            <span>{a.text}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** One zone. Clicking anywhere on the card opens the zone's details. */
export default function ZoneCard({ zone, now, connected, onSelect, countsAvailable = true }) {
  const state = zoneState(zone, now, connected);
  const level = displayLevel(zone, now, connected);
  const reading = zone.reading;
  const live = state === "live";

  return (
    <article
      className={`relative rounded-xl border-4 p-4 shadow-sm transition-colors ${
        live ? `bg-white ${level.border}` : "border-gray-300 bg-gray-200 text-gray-700"
      }`}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h3 className="text-lg font-bold leading-snug">
          <button
            type="button"
            onClick={() => onSelect(zone.id)}
            className="text-left after:absolute after:inset-0 after:rounded-xl after:content-[''] focus:outline-none focus-visible:after:ring-4 focus-visible:after:ring-sky-600"
          >
            {zone.name}
          </button>
        </h3>
        <RiskBadge level={level} />
      </div>
      {zone.kind === "video" && (
        <p className="mt-1 inline-block rounded bg-violet-800 px-2 py-0.5 text-xs font-bold uppercase tracking-wide text-white">
          Test video, not a live camera
        </p>
      )}

      {live ? (
        <>
          <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2">
            <Metric label="People" value={formatCount(reading.count)} />
            <Metric label="Density" value={`${formatNumber(reading.density)} /m²`} />
            <Metric label="Turbulence" value={`${formatNumber(reading.turbulence)} (${turbulenceLabel(reading.turbulence)})`} />
            <Metric label="Crowd pressure" value={formatNumber(reading.pressure)} />
          </dl>
          {reading.count_method && (
            <p className="mt-2 text-sm text-slate-700">
              Counted by <span className="font-bold text-slate-900">{countMethodLabel(reading.count_method)}</span>
              {reading.count_method === "density" && reading.detection_count != null &&
                ` (YOLO found ${formatCount(reading.detection_count)})`}
            </p>
          )}
          {reading.hotspot && (
            <p className="mt-1 text-sm text-slate-700">
              <span className="font-bold text-slate-900">Hotspot:</span> {hotspotText(reading.hotspot)}
            </p>
          )}
          <GateActions actions={reading.gate_actions} level={level} />
          <Exits exits={reading.exits} />
        </>
      ) : (
        <p className="mt-3 font-semibold">
          {!countsAvailable
            ? "No trained model: counts unavailable."
            : state === "waiting"
            ? "Waiting for the first reading…"
            : reading
              ? `No signal since ${formatTime(zone.lastSeen)}. Risk level unknown.`
              : "No signal. Risk level unknown."}
        </p>
      )}
    </article>
  );
}
