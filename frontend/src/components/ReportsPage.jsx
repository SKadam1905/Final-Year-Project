import { useCallback, useEffect, useState } from "react";
import { reportsApi } from "../api.js";
import { formatCount, formatNumber, formatTime } from "../format.js";
import {
  ALERT_COLUMNS, INCIDENT_COLUMNS, LEVELS, PRESETS, SUMMARY_COLUMNS, downloadCsv, epochToLocal, formatSpan,
  reportQuery, timeRange, toCsv,
} from "../reports.js";
import { riskStyle } from "../risk.js";
import RiskBadge from "./RiskBadge.jsx";

const BUTTON = "rounded-md px-3 py-2 text-sm font-bold focus:outline-none focus-visible:ring-4 focus-visible:ring-sky-500";
const TH = "border-b-2 border-slate-300 px-2 py-1.5 text-left text-xs font-semibold uppercase tracking-wide text-slate-700";
const TD = "border-b border-slate-200 px-2 py-1.5 align-top tabular-nums";

const dateTime = new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeStyle: "medium", hour12: false });
const when = (epochSeconds) => (epochSeconds == null ? "—" : dateTime.format(new Date(epochSeconds * 1000)));
const whenIso = (iso) => (iso ? dateTime.format(new Date(iso)) : "ongoing");
const stamp = () => new Date().toISOString().slice(0, 16).replace(/\D/g, ""); // e.g. 202610091405

function Section({ id, title, count, onExport, children }) {
  return (
    <section aria-labelledby={id} className="mt-6 break-inside-avoid">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 id={id} className="text-lg font-bold">{title} <span className="font-normal text-slate-700">({count})</span></h3>
        {onExport && (
          <button type="button" onClick={onExport} disabled={!count}
            className={`${BUTTON} border-2 border-slate-900 bg-white text-slate-900 disabled:opacity-40 print:hidden`}>
            Export CSV
          </button>
        )}
      </div>
      <div className="mt-2 overflow-x-auto">{children}</div>
    </section>
  );
}

function SummaryTable({ zones }) {
  if (!zones.length) return <p className="text-slate-700">No readings in this range.</p>;
  return (
    <table className="w-full min-w-[48rem] text-sm">
      <thead>
        <tr>
          <th className={TH}>Zone</th><th className={TH}>Readings</th><th className={TH}>Peak people</th>
          <th className={TH}>Peak density</th><th className={TH}>Peak pressure</th><th className={TH}>Highest level</th>
          {LEVELS.map((level) => <th key={level} className={TH}>Time {level}</th>)}
        </tr>
      </thead>
      <tbody>
        {zones.map((z) => (
          <tr key={z.zone_id}>
            <td className={`${TD} font-semibold`}>{z.name}</td>
            <td className={TD}>{z.readings}</td>
            <td className={TD}>{formatCount(z.peak_count)} <span className="block text-xs text-slate-600">{formatTime(z.peak_count_time * 1000)}</span></td>
            <td className={TD}>{formatNumber(z.peak_density)} /m² <span className="block text-xs text-slate-600">{formatTime(z.peak_density_time * 1000)}</span></td>
            <td className={TD}>{formatNumber(z.peak_pressure)}</td>
            <td className={TD}><RiskBadge level={riskStyle(z.highest_level)} size="sm" /></td>
            {LEVELS.map((level) => <td key={level} className={TD}>{formatSpan(z.time_at_level_s[level])}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function AlertsTable({ alerts, zoneNames }) {
  if (!alerts.length) return <p className="text-slate-700">No alerts match these filters.</p>;
  return (
    <table className="w-full min-w-[40rem] text-sm">
      <thead>
        <tr><th className={TH}>Time</th><th className={TH}>Zone</th><th className={TH}>Level</th><th className={TH}>Message</th><th className={TH}>Delivered</th></tr>
      </thead>
      <tbody>
        {alerts.map((a) => (
          <tr key={a.id ?? `${a.zone_id}@${a.timestamp}`}>
            <td className={`${TD} whitespace-nowrap`}>{when(a.timestamp)}</td>
            <td className={TD}>{zoneNames[a.zone_id] ?? a.zone_id}</td>
            <td className={TD}><RiskBadge level={riskStyle(a.risk_level)} size="sm" /></td>
            <td className={`${TD} whitespace-pre-line`}>{a.message}</td>
            <td className={TD}>{a.delivered ? "Yes" : "No"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function IncidentsTable({ incidents, zoneNames }) {
  if (!incidents.length) return <p className="text-slate-700">No ORANGE or RED periods match these filters.</p>;
  return (
    <table className="w-full min-w-[40rem] text-sm">
      <thead>
        <tr><th className={TH}>Zone</th><th className={TH}>Started</th><th className={TH}>Ended</th><th className={TH}>Peak level</th>
          <th className={TH}>Peak pressure</th><th className={TH}>Peak people</th></tr>
      </thead>
      <tbody>
        {incidents.map((i) => (
          <tr key={i.id}>
            <td className={TD}>{zoneNames[i.zone_id] ?? i.zone_id}</td>
            <td className={`${TD} whitespace-nowrap`}>{whenIso(i.started_at)}</td>
            <td className={`${TD} whitespace-nowrap`}>{whenIso(i.ended_at)}</td>
            <td className={TD}><RiskBadge level={riskStyle(i.peak_level)} size="sm" /></td>
            <td className={TD}>{formatNumber(i.peak_pressure)}</td>
            <td className={TD}>{formatCount(i.peak_count)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Reports: filterable alert and incident history, per-zone summary, CSV export, printable. */
export default function ReportsPage({ zones, api = reportsApi }) {
  const zoneNames = Object.fromEntries(zones.map((z) => [z.id, z.name]));
  const [filters, setFilters] = useState(() => ({
    zoneId: "", levels: ["ORANGE", "RED"], preset: "1h",
    from: epochToLocal(Date.now() - 3600_000), to: epochToLocal(Date.now()),
  }));
  const [result, setResult] = useState({ status: "idle", error: null, summary: null, alerts: [], incidents: [], range: null });

  const run = useCallback(async (f) => {
    const range = timeRange(f.preset, f.from, f.to);
    const query = { zoneId: f.zoneId, levels: f.levels, ...range };
    setResult((r) => ({ ...r, status: "loading", error: null }));
    try {
      const [summary, alerts, incidents] = await Promise.all([
        api.summary(reportQuery(query, false)), api.alerts(reportQuery(query)), api.incidents(reportQuery(query)),
      ]);
      setResult({ status: "done", error: null, summary, alerts, incidents, range });
    } catch (error) {
      setResult((r) => ({ ...r, status: "error", error: error.message }));
    }
  }, [api]);

  useEffect(() => {
    run(filters);
  }, [run]); // first load only: later reports run when "Show report" is pressed

  const set = (patch) => setFilters((f) => ({ ...f, ...patch }));
  const toggleLevel = (level) => set({
    levels: filters.levels.includes(level) ? filters.levels.filter((l) => l !== level) : [...filters.levels, level],
  });
  const describe = result.range
    ? `${when(result.range.since)} to ${when(result.range.until)}` : "";
  const zoneLabel = filters.zoneId ? zoneNames[filters.zoneId] ?? filters.zoneId : "All zones";

  return (
    <main className="mx-auto max-w-7xl px-3 py-4 sm:px-6">
      <h2 className="text-2xl font-bold">Reports</h2>
      <p className="hidden text-sm print:block">
        Ramkund crowd safety · {zoneLabel} · {describe} · levels {filters.levels.join(", ") || "all"} · printed {when(Date.now() / 1000)}
      </p>

      <form className="mt-3 grid gap-3 rounded-xl bg-white p-4 shadow-sm sm:grid-cols-2 lg:grid-cols-4 print:hidden"
        onSubmit={(event) => { event.preventDefault(); run(filters); }}>
        <label className="text-sm font-semibold">
          Zone
          <select value={filters.zoneId} onChange={(e) => set({ zoneId: e.target.value })}
            className="mt-1 block w-full rounded-md border-2 border-slate-400 bg-white px-2 py-1.5">
            <option value="">All zones</option>
            {zones.map((z) => <option key={z.id} value={z.id}>{z.name}</option>)}
          </select>
        </label>
        <label className="text-sm font-semibold">
          Time range
          <select value={filters.preset} onChange={(e) => set({ preset: e.target.value })}
            className="mt-1 block w-full rounded-md border-2 border-slate-400 bg-white px-2 py-1.5">
            {PRESETS.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
          </select>
        </label>
        {filters.preset === "custom" ? (
          <div className="grid grid-cols-2 gap-2 sm:col-span-2 lg:col-span-1">
            <label className="text-sm font-semibold">From
              <input type="datetime-local" value={filters.from} onChange={(e) => set({ from: e.target.value })}
                className="mt-1 block w-full rounded-md border-2 border-slate-400 px-1 py-1.5 text-sm" />
            </label>
            <label className="text-sm font-semibold">To
              <input type="datetime-local" value={filters.to} onChange={(e) => set({ to: e.target.value })}
                className="mt-1 block w-full rounded-md border-2 border-slate-400 px-1 py-1.5 text-sm" />
            </label>
          </div>
        ) : <div className="hidden lg:block" />}
        <fieldset className="text-sm font-semibold">
          <legend>Risk levels (alerts and incidents)</legend>
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1">
            {LEVELS.map((level) => (
              <label key={level} className="flex items-center gap-1 font-normal">
                <input type="checkbox" checked={filters.levels.includes(level)} onChange={() => toggleLevel(level)} className="h-4 w-4" />
                {level}
              </label>
            ))}
          </div>
        </fieldset>
        <div className="flex flex-wrap items-end gap-2 sm:col-span-2 lg:col-span-4">
          <button type="submit" className={`${BUTTON} bg-slate-900 text-white`}>Show report</button>
          <button type="button" onClick={() => window.print()} className={`${BUTTON} border-2 border-slate-900 bg-white text-slate-900`}>
            Print
          </button>
          <span role="status" className="text-sm text-slate-700">
            {result.status === "loading" ? "Loading…" : result.status === "done" ? `${zoneLabel}, ${describe}` : ""}
          </span>
        </div>
      </form>

      {result.status === "error" && (
        <p role="alert" className="mt-4 rounded-md bg-red-100 p-3 font-semibold text-red-900">Report unavailable: {result.error}</p>
      )}

      {result.summary && (
        <>
          <Section id="summary-heading" title="Per-zone summary" count={result.summary.zones.length}
            onExport={() => downloadCsv(`zone_summary_${stamp()}.csv`, toCsv(result.summary.zones, SUMMARY_COLUMNS))}>
            <SummaryTable zones={result.summary.zones} />
            <p className="mt-1 text-xs text-slate-700">
              From {result.summary.readings} stored readings ({result.summary.storage === "memory"
                ? "in memory: only the newest 5000 readings, lost on restart" : "MongoDB"}). Time at a level counts each
              reading until the next one, at most 10 s; longer gaps are outages and are not counted.
            </p>
          </Section>
          <Section id="alerts-heading" title="Alert history" count={result.alerts.length}
            onExport={() => downloadCsv(`alerts_${stamp()}.csv`, toCsv(result.alerts, ALERT_COLUMNS))}>
            <AlertsTable alerts={result.alerts} zoneNames={zoneNames} />
          </Section>
          <Section id="incidents-heading" title="Incidents (ORANGE/RED periods)" count={result.incidents.length}
            onExport={() => downloadCsv(`incidents_${stamp()}.csv`, toCsv(result.incidents, INCIDENT_COLUMNS))}>
            <IncidentsTable incidents={result.incidents} zoneNames={zoneNames} />
          </Section>
        </>
      )}
    </main>
  );
}
