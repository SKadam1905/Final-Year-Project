import { levelUi, modelName } from "./theme.js";
import { useAnimatedNumber } from "./useAnimatedNumber.js";

function Count({ value }) {
  const shown = useAnimatedNumber(value);
  return value == null ? "—" : Math.round(shown).toLocaleString("en-IN");
}

const COLOURS = { saffron: "#F2A54A", neutral: "#3B5182", quiet: "#2A3D66", red: "#EF4444", orange: "#FB923C", green: "#22C55E" };

function Tile({ label, children, sub, accent = COLOURS.quiet }) {
  return (
    <div className="min-w-0 rounded-xl border-l-4 bg-navy-800 px-3 py-2" style={{ borderLeftColor: accent }}>
      <p className="truncate text-[11px] font-semibold uppercase tracking-wider text-slate-400">{label}</p>
      <div className="mt-0.5 font-display text-2xl font-semibold leading-tight text-white tabular-nums sm:text-[1.7rem]">{children}</div>
      {sub && <p className="truncate text-xs text-slate-400">{sub}</p>}
    </div>
  );
}

const LEVELS = ["RED", "ORANGE", "YELLOW", "GREEN"];

/** Live KPIs: every value comes from the backend; "—" where there is no data. */
export default function KpiStrip({ kpis, health }) {
  const top = kpis.highest;
  const topUi = top ? levelUi(top.level) : null;
  return (
    <section aria-label="Key figures" className="grid grid-cols-2 gap-2 px-3 sm:grid-cols-3 sm:px-4 lg:grid-cols-6">
      <Tile label="People now" accent={COLOURS.saffron}
        sub={kpis.liveZones ? `${kpis.liveZones} live camera${kpis.liveZones === 1 ? "" : "s"}${kpis.testZones ? `, incl. ${kpis.testZones} test video` : ""}` : "no live camera"}>
        <span aria-live="off"><Count value={kpis.people} /></span>
      </Tile>
      <Tile label="Zones by risk" accent={COLOURS.neutral}>
        <ul className="flex flex-wrap gap-1.5 text-base">
          {LEVELS.map((level) => {
            const ui = levelUi(level);
            return (
              <li key={level} title={`${kpis.byLevel[level]} at ${level}`}
                className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-sm font-semibold ${kpis.byLevel[level] ? ui.chip : "bg-navy-700 text-slate-400"}`}>
                <span aria-hidden="true">{ui.icon}</span>{kpis.byLevel[level]}<span className="sr-only"> {level}</span>
              </li>
            );
          })}
        </ul>
        {kpis.byLevel["NO SIGNAL"] > 0 && <p className="mt-0.5 text-xs font-normal text-slate-300">✕ {kpis.byLevel["NO SIGNAL"]} no signal</p>}
      </Tile>
      <Tile label="Highest risk now" accent={top ? topUi.fill : COLOURS.quiet} sub={top ? top.zone.name : "no live reading"}>
        {top ? (
          <span className={`inline-flex items-center gap-2 rounded-md px-2 py-0.5 text-xl ${topUi.chip} ${top.level === "RED" ? "animate-pulse motion-safe-only" : ""}`}>
            <span aria-hidden="true">{topUi.icon}</span>{top.level}
          </span>
        ) : "—"}
      </Tile>
      <Tile label="Unacknowledged alerts" accent={kpis.unacknowledged ? COLOURS.red : COLOURS.quiet}
        sub={kpis.unacknowledged ? "need a response" : "none waiting"}>
        <span className={kpis.unacknowledged ? "text-red-300" : ""}><Count value={kpis.unacknowledged} /></span>
      </Tile>
      <Tile label="Cameras online" accent={kpis.camerasOnline < kpis.camerasTotal ? COLOURS.orange : COLOURS.green}
        sub={health ? `${kpis.camerasTotal - kpis.camerasOnline} offline` : "status unknown"}>
        {health ? <><Count value={kpis.camerasOnline} /><span className="text-lg text-slate-400"> / {kpis.camerasTotal}</span></> : "—"}
      </Tile>
      <Tile label="Active model" accent={COLOURS.neutral}
        sub={health?.model?.test_mae ? `test MAE ${Number(health.model.test_mae).toFixed(1)} (ShanghaiTech A)` : health?.model_error ? "counts unavailable" : ""}>
        <span className="block text-sm font-medium leading-5" title={health?.model?.label}>{modelName(health)}</span>
      </Tile>
    </section>
  );
}
