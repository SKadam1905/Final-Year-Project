import { formatCount, formatDuration, formatNumber } from "../format.js";
import { LEVELS, riskStyle } from "../risk.js";
import RiskBadge from "./RiskBadge.jsx";

const LEVEL_ORDER = ["RED", "ORANGE", "YELLOW", "GREEN"];
const BAR = { RED: "bg-red-700", ORANGE: "bg-orange-500", YELLOW: "bg-yellow-400", GREEN: "bg-green-700" };

function Tile({ label, value, detail }) {
  return (
    <div className="rounded-lg bg-slate-50 p-3 ring-1 ring-slate-200">
      <dt className="text-xs font-semibold uppercase tracking-wide text-slate-600">{label}</dt>
      <dd className="text-2xl font-bold tabular-nums text-slate-900">{value}</dd>
      {detail && <dd className="text-sm text-slate-700">{detail}</dd>}
    </div>
  );
}

/** The headline numbers of an analysed video (`summary` from GET /api/jobs/{id}/results). */
export default function ReportSummary({ summary }) {
  const total = Object.values(summary.time_at_level_s).reduce((a, b) => a + b, 0) || 1;
  return (
    <section aria-labelledby="summary-heading">
      <h3 id="summary-heading" className="text-lg font-bold">Summary</h3>
      <dl className="mt-2 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <Tile label="Video length" value={formatDuration(summary.duration_s)}
          detail={`${summary.samples} samples analysed, one every 2 s`} />
        <Tile label="Peak count" value={`${formatCount(summary.peak_count)} people`}
          detail={`at ${formatDuration(summary.peak_count_time_s)}`} />
        <Tile label="Peak density" value={`${formatNumber(summary.peak_density)} /m²`}
          detail={`at ${formatDuration(summary.peak_density_time_s)}`} />
        <Tile label="Peak crowd pressure" value={formatNumber(summary.peak_pressure)}
          detail={`30 s mean, at ${formatDuration(summary.peak_pressure_time_s)}`} />
        <div className="rounded-lg bg-slate-50 p-3 ring-1 ring-slate-200">
          <dt className="text-xs font-semibold uppercase tracking-wide text-slate-600">Highest risk level</dt>
          <dd className="mt-1"><RiskBadge level={riskStyle(summary.highest_level)} /></dd>
        </div>
      </dl>

      <h4 className="mt-4 font-bold">Time at each risk level</h4>
      <div className="mt-2 flex h-4 overflow-hidden rounded-full ring-1 ring-slate-300" aria-hidden="true">
        {LEVEL_ORDER.map((level) => (
          <div key={level} className={BAR[level]} style={{ width: `${(100 * summary.time_at_level_s[level]) / total}%` }} />
        ))}
      </div>
      <ul className="mt-2 grid gap-1 text-sm sm:grid-cols-4">
        {LEVEL_ORDER.map((level) => (
          <li key={level} className="font-semibold">
            <span aria-hidden="true">{LEVELS[level].icon} </span>
            {level}: <span className="tabular-nums">{formatDuration(summary.time_at_level_s[level])}</span>
            <span className="font-normal text-slate-700"> ({Math.round((100 * summary.time_at_level_s[level]) / total)}%)</span>
          </li>
        ))}
      </ul>

      {(summary.scene_cut_samples > 0 || summary.moving_camera_samples > 0) && (
        <ul className="mt-3 space-y-1 text-sm text-slate-800">
          {summary.scene_cut_samples > 0 && (
            <li>
              <span className="font-bold">Scene cuts:</span> {summary.scene_cut_samples} samples spanned a change of shot
              ({formatDuration(summary.time_scene_cut_s)} of video). They were skipped and are not in the numbers above.
            </li>
          )}
          {summary.moving_camera_samples > 0 && (
            <li>
              <span className="font-bold">Moving camera:</span> {summary.moving_camera_samples} samples. The camera's own
              motion was removed before measuring turbulence.
            </li>
          )}
        </ul>
      )}
    </section>
  );
}
