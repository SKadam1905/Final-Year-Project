import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatTime } from "../format.js";

export const WINDOW_SECONDS = 300;

const LINES = [
  { key: "yellow", name: "Yellow", stroke: "#a16207" },
  { key: "orange", name: "Orange", stroke: "#c2410c" },
  { key: "red", name: "Red", stroke: "#b91c1c" },
];

/** Crowd pressure (30 s mean) over the last five minutes, with the risk thresholds marked. */
export default function PressureChart({ points, thresholds }) {
  const latest = points.at(-1)?.t;
  const data = latest == null ? [] : points.filter((p) => p.t >= latest - WINDOW_SECONDS).map((p) => ({
    time: p.t * 1000,
    pressure: Math.max(0, p.pressure),
  }));
  if (data.length === 0) return <p className="text-sm text-slate-700">No readings yet for the chart.</p>;

  const top = Math.ceil(Math.max((thresholds?.red ?? 0) * 1.25, ...data.map((d) => d.pressure), 1));
  return (
    <figure>
      <figcaption className="mb-2 text-sm font-semibold text-slate-800">
        Crowd pressure, last {WINDOW_SECONDS / 60} minutes
        {thresholds && ` (thresholds: yellow ${thresholds.yellow}, orange ${thresholds.orange}, red ${thresholds.red})`}
      </figcaption>
      <div className="h-60 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
            <XAxis dataKey="time" type="number" scale="time" domain={["dataMin", "dataMax"]}
              tickFormatter={formatTime} minTickGap={40} tick={{ fill: "#1e293b", fontSize: 12 }} />
            <YAxis domain={[0, top]} allowDecimals={false} tick={{ fill: "#1e293b", fontSize: 12 }} />
            <Tooltip labelFormatter={formatTime} formatter={(value) => [value.toFixed(2), "Pressure"]} />
            {thresholds && LINES.map((line) => (
              <ReferenceLine key={line.key} y={thresholds[line.key]} stroke={line.stroke} strokeWidth={2}
                strokeDasharray="6 4" ifOverflow="extendDomain"
                label={{ value: `${line.name} ${thresholds[line.key]}`, position: "insideTopLeft", fill: line.stroke,
                  fontSize: 12, fontWeight: 700 }} />
            ))}
            <Line type="monotone" dataKey="pressure" stroke="#0f172a" strokeWidth={2.5} dot={false}
              isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
