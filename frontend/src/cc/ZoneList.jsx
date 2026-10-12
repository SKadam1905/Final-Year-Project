import { formatCount, formatNumber } from "../format.js";
import { SURGE_TONES, displayLevel, surgeStatus, surgeRuleText, zoneState } from "../risk.js";
import { levelUi } from "./theme.js";

/** Every camera zone as a compact, selectable row: level (word + icon), people, density, pressure, speed surge. */
export default function ZoneList({ zones, now, connected, selectedId, onSelect, title = "All cameras", surge }) {
  return (
    <section aria-labelledby="zone-list-heading" className="rounded-xl bg-navy-800 p-3">
      <h2 id="zone-list-heading" className="font-display text-sm font-semibold uppercase tracking-wider text-slate-300">{title}</h2>
      {zones.length === 0 ? (
        <p className="mt-2 text-sm text-slate-400">No cameras are configured on the backend (config/zones.json).</p>
      ) : (
        <ul className="mt-2 grid gap-1.5 sm:grid-cols-2 2xl:grid-cols-3">
          {zones.map((zone) => {
            const level = displayLevel(zone, now, connected).label;
            const ui = levelUi(level);
            const live = zoneState(zone, now, connected) === "live";
            const selected = zone.id === selectedId;
            const speed = live ? surgeStatus(zone.reading, surge) : null;
            return (
              <li key={zone.id}>
                <button type="button" onClick={() => onSelect(zone.id)} aria-pressed={selected}
                  className={`flex w-full items-center gap-2 rounded-lg border px-2.5 py-2 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-saffron ${
                    selected ? "border-saffron bg-navy-700" : "border-navy-600 bg-navy-900 hover:border-navy-500"}`}>
                  <span className={`inline-flex w-[5.5rem] shrink-0 items-center justify-center gap-1 rounded px-1.5 py-0.5 font-display text-xs font-semibold ${ui.chip}`}>
                    <span aria-hidden="true">{ui.icon}</span>{ui.word}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-semibold text-white">
                      {zone.name}{zone.kind === "video" && <span className="ml-1 rounded bg-violet-700 px-1 text-[10px] font-bold uppercase text-white">test video</span>}
                    </span>
                    <span className="block text-xs text-slate-400 tabular-nums">
                      {live ? `${formatCount(zone.reading.count)} people · ${formatNumber(zone.reading.density)} /m² · pressure ${formatNumber(zone.reading.pressure)}`
                        : level === "WAITING" ? "waiting for the first reading" : "no signal: values unknown"}
                      {speed && (
                        <span className={SURGE_TONES[speed.tone]} title={`Speed surge: ${speed.note}. ${surgeRuleText(surge)}`}>
                          {" · surge "}{speed.text}{speed.tone === "fired" || speed.tone === "raised" ? " ⚡" : ""}
                        </span>
                      )}
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
