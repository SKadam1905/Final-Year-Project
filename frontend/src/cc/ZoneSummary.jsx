import { formatCount, formatNumber } from "../format.js";
import { SURGE_TONES, displayLevel, surgeStatus, surgeRuleText, turbulenceLabel, zoneState } from "../risk.js";
import { levelUi } from "./theme.js";

function Figure({ label, value, unit }) {
  return (
    <div>
      <dt className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">{label}</dt>
      <dd className="font-display text-2xl font-semibold text-white tabular-nums">{value}{unit && <span className="ml-1 text-sm text-slate-400">{unit}</span>}</dd>
    </div>
  );
}

/** Speed-surge indicator: the ratio, what it means, and whether the trigger can raise the level. */
function SurgeLine({ reading, surge }) {
  const s = surgeStatus(reading, surge);
  return (
    <p className="mt-2 text-xs text-slate-400" title={surgeRuleText(surge)}>
      <span className="font-semibold uppercase tracking-wider">Speed surge </span>
      <span className={`tabular-nums ${SURGE_TONES[s.tone]}`}>{s.text}{s.tone === "fired" || s.tone === "raised" ? " ⚡" : ""}</span>
      <span> · {s.note}{surge && !surge.trigger ? " · trigger off" : ""}</span>
    </p>
  );
}

/** The selected zone at a glance: level (word + icon + colour) and the four live figures. */
export default function ZoneSummary({ zone, now, connected, surge }) {
  if (!zone) {
    return <section className="rounded-xl bg-navy-800 p-4 text-sm text-slate-400">Select a zone to see its details.</section>;
  }
  const level = displayLevel(zone, now, connected).label;
  const ui = levelUi(level);
  const live = zoneState(zone, now, connected) === "live";
  const r = zone.reading;
  return (
    <section aria-labelledby="selected-zone-heading" className="rounded-xl border border-navy-600 bg-navy-800 p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-saffron">Selected zone</p>
          <h2 id="selected-zone-heading" className="font-display text-xl font-semibold text-white">{zone.name}</h2>
          {zone.kind === "video" && <p className="mt-0.5 inline-block rounded bg-violet-700 px-1.5 text-[11px] font-bold uppercase text-white">Test video, not a live camera</p>}
        </div>
        <span className={`relative inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 font-display text-lg font-bold ${ui.chip}`}>
          {(level === "RED" || level === "ORANGE") && (
            <span aria-hidden="true" className={`absolute inset-0 rounded-lg motion-safe-only ${level === "RED" ? "animate-chip-pulse-fast" : "animate-chip-pulse"}`}
              style={{ boxShadow: `0 0 0 3px ${ui.fill}` }} />
          )}
          <span aria-hidden="true">{ui.icon}</span>{ui.word}<span className="text-xs font-semibold opacity-80">{ui.meaning}</span>
        </span>
      </div>
      {live ? (
        <>
          <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2">
            <Figure label="People" value={formatCount(r.count)} />
            <Figure label="Density" value={formatNumber(r.density)} unit="/m²" />
            <Figure label="Turbulence" value={formatNumber(r.turbulence)} unit={turbulenceLabel(r.turbulence)} />
            <Figure label="Crowd pressure" value={formatNumber(r.pressure)} unit="30 s mean" />
          </dl>
          <SurgeLine reading={r} surge={surge} />
        </>
      ) : (
        <p className="mt-3 text-sm font-medium text-slate-300">
          {level === "WAITING" ? "Waiting for the first reading from this camera…" : "No signal: no reading in the last 10 s, so its risk is unknown."}
        </p>
      )}
    </section>
  );
}
