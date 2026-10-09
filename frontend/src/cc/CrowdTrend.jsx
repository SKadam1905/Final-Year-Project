import {
  CartesianGrid, ComposedChart, Legend, Line, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { formatCount } from "../format.js";

const WINDOW_S = 600;
const hhmm = new Intl.DateTimeFormat("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false });
const BANDS = [
  { key: "green", from: (t) => 0, to: (t) => t.yellow, fill: "#22C55E" },
  { key: "yellow", from: (t) => t.yellow, to: (t) => t.orange, fill: "#FACC15" },
  { key: "orange", from: (t) => t.orange, to: (t) => t.red, fill: "#FB923C" },
  { key: "red", from: (t) => t.red, to: (t, top) => top, fill: "#EF4444" },
];
const TICK = { fill: "#CBD5E1", fontSize: 11 };

/** The last 10 minutes of one zone: density (left axis, on the risk bands) and people (right axis). */
export default function CrowdTrend({ history, thresholds }) {
  const latest = history.at(-1)?.t;
  const data = history.filter((p) => p.t >= latest - WINDOW_S && p.density != null)
    .map((p) => ({ time: p.t * 1000, density: p.density, count: p.count }));
  if (data.length < 2 || !thresholds) {
    return <p className="grid h-40 place-items-center text-sm text-slate-400">Collecting readings: the chart fills in over the next minute.</p>;
  }
  const top = Math.max(thresholds.red + 1, ...data.map((d) => d.density));
  return (
    <figure>
      <figcaption className="text-xs text-slate-400">
        Last 10 minutes: density on the risk bands ({thresholds.yellow} / {thresholds.orange} / {thresholds.red} per m²) and people counted
      </figcaption>
      <div className="h-44 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 6, right: 0, bottom: 0, left: -18 }}>
            {BANDS.map((b) => (
              <ReferenceArea key={b.key} yAxisId="density" y1={b.from(thresholds)} y2={b.to(thresholds, top)} fill={b.fill}
                fillOpacity={0.12} strokeOpacity={0} ifOverflow="hidden" />
            ))}
            <CartesianGrid strokeDasharray="3 3" stroke="#1D2D50" />
            <XAxis dataKey="time" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(t) => hhmm.format(t)} tick={TICK} minTickGap={40} />
            <YAxis yAxisId="density" domain={[0, Math.ceil(top)]} tick={TICK} allowDecimals={false} />
            <YAxis yAxisId="count" orientation="right" tick={TICK} allowDecimals={false} width={44} />
            <Tooltip contentStyle={{ background: "#0B1220", border: "1px solid #3B5182", fontSize: 12 }} labelStyle={{ color: "#F1F5F9" }}
              labelFormatter={(t) => hhmm.format(t)}
              formatter={(value, name) => [name === "People" ? formatCount(value) : `${Number(value).toFixed(2)} /m²`, name]} />
            <Legend wrapperStyle={{ fontSize: 12, color: "#CBD5E1" }} />
            <Line yAxisId="density" name="Density" dataKey="density" stroke="#F2A54A" strokeWidth={2.5} dot={false} isAnimationActive={false} />
            <Line yAxisId="count" name="People" dataKey="count" stroke="#93C5FD" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
