import { useEffect, useState } from "react";
import { api } from "../api.js";
import { formatCount, formatNumber } from "../format.js";
import { turbulenceLabel, zoneState } from "../risk.js";
import { buildMapModel } from "./mapModel.js";
import { levelUi } from "./theme.js";
import { useAnimatedNumber } from "./useAnimatedNumber.js";

const WALKWAY = { // band -> stroke colour and width: busier walkways are drawn thicker and warmer
  GREEN: { stroke: "#64748B", width: 4 },
  YELLOW: { stroke: "#FACC15", width: 6 },
  ORANGE: { stroke: "#FB923C", width: 8 },
  RED: { stroke: "#EF4444", width: 11 },
};
const R = { zone: 36, ghat: 36, gate: 27 };

function AnimatedCount({ value }) {
  const shown = useAnimatedNumber(value);
  return formatCount(shown ?? 0);
}

function Tooltip({ node, now, connected, width, height }) {
  const left = `${(node.x / width) * 100}%`;
  const below = node.y < height * 0.35;
  return (
    <div role="tooltip" className="pointer-events-none absolute z-10 w-60 -translate-x-1/2 animate-fade-in rounded-lg border border-navy-500 bg-navy-950/95 p-3 text-sm shadow-xl"
      style={{ left, top: below ? `${((node.y + 40) / height) * 100}%` : undefined, bottom: below ? undefined : `${((height - node.y + 40) / height) * 100}%` }}>
      <p className="font-display font-semibold text-white">{node.label}</p>
      {node.zones.map((zone) => {
        const live = zoneState(zone, now, connected) === "live";
        const r = zone.reading;
        return (
          <div key={zone.id} className="mt-1.5 border-t border-navy-700 pt-1.5">
            <p className="text-xs text-slate-300">{zone.name}{zone.kind === "video" && " · test video"}</p>
            {live ? (
              <dl className="mt-1 grid grid-cols-2 gap-x-3 text-xs tabular-nums">
                <dt className="text-slate-400">People</dt><dd className="text-right text-white">{formatCount(r.count)}</dd>
                <dt className="text-slate-400">Density</dt><dd className="text-right text-white">{formatNumber(r.density)} /m²</dd>
                <dt className="text-slate-400">Turbulence</dt><dd className="text-right text-white">{formatNumber(r.turbulence)} ({turbulenceLabel(r.turbulence)})</dd>
                <dt className="text-slate-400">Pressure</dt><dd className="text-right text-white">{formatNumber(r.pressure)}</dd>
                <dt className="text-slate-400">Risk</dt><dd className="text-right font-semibold text-white">{r.risk_level}</dd>
              </dl>
            ) : <p className="text-xs text-slate-400">No signal: no recent reading.</p>}
          </div>
        );
      })}
      {!node.zones.length && <p className="mt-1 text-xs text-slate-400">{node.kind === "gate" ? "Exit gate" : "No camera here"}</p>}
    </div>
  );
}

function Node({ node, selected, safestExit, onSelect, onHover }) {
  const { x, y } = node;
  const r = R[node.kind] ?? 26;
  const ui = node.level ? levelUi(node.level) : null;
  const hasCamera = node.zones.length > 0;
  const label = { // name and level text, on the side config/site_map.json asks for (label_at)
    below: { x, y1: y + r + 26, y2: y + r + 48, anchor: "middle" },
    above: { x, y1: y - r - 34, y2: y - r - 12, anchor: "middle" },
    left: { x: x - r - 14, y1: y - 3, y2: y + 20, anchor: "end" },
    right: { x: x + r + 14, y1: y - 3, y2: y + 20, anchor: "start" },
  }[node.label_at ?? "below"] ?? { x, y1: y + r + 26, y2: y + r + 48, anchor: "middle" };
  const interactive = hasCamera ? {
    role: "button", tabIndex: 0, "aria-pressed": selected,
    "aria-label": `${node.label}: ${node.level}${node.live ? `, ${formatCount(node.primary.reading.count)} people` : ""}. Select for details.`,
    onClick: () => onSelect(node.primary.id),
    onKeyDown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(node.primary.id); } },
    className: "cursor-pointer outline-none [&:focus-visible>.focus-ring]:opacity-100",
  } : { "aria-label": `${node.label}${node.kind === "gate" ? " (exit gate)" : ", no camera"}`, role: "img" };

  return (
    <g {...interactive} onMouseEnter={() => onHover(node.id)} onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(node.id)} onBlur={() => onHover(null)}>
      {(node.level === "RED" || node.level === "ORANGE") && (
        <circle cx={x} cy={y} r={r} fill={ui.fill} opacity="0.6" style={{ transformBox: "fill-box", transformOrigin: "center" }}
          className={`motion-safe-only ${node.level === "RED" ? "animate-risk-pulse-fast" : "animate-risk-pulse"}`} />
      )}
      {safestExit && (
        <circle cx={x} cy={y} r={r + 12} fill="none" stroke="#F2A54A" strokeWidth="3" strokeDasharray="6 5"
          style={{ transformBox: "fill-box", transformOrigin: "center" }} className="motion-safe-only animate-[spin_6s_linear_infinite]" />
      )}
      <circle className="focus-ring opacity-0 transition" cx={x} cy={y} r={r + 7} fill="none" stroke="#F8C98F" strokeWidth="3" />
      {selected && <circle cx={x} cy={y} r={r + 6} fill="none" stroke="#F2A54A" strokeWidth="4" />}
      {node.kind === "gate" ? (
        <>
          <rect x={x - r} y={y - r} width={r * 2} height={r * 2} rx="7" fill={safestExit ? "#F2A54A" : "#1D2D50"}
            stroke={safestExit ? "#F8C98F" : "#3B5182"} strokeWidth="2.5" />
          <text x={x} y={y + 9} textAnchor="middle" className="font-display" fontSize="26" fontWeight="700"
            fill={safestExit ? "#0B1220" : "#E2E8F0"}>{node.label.replace(/^Gate /, "")}</text>
        </>
      ) : hasCamera ? (
        <>
          <circle cx={x} cy={y} r={r} fill={ui.fill} stroke="#0B1220" strokeWidth="3" />
          <text x={x} y={y + 8} textAnchor="middle" className="font-display tabular-nums" fontSize={node.live ? 22 : 20} fontWeight="700"
            fill={["RED", "NO SIGNAL", "WAITING"].includes(node.level) ? "#FFFFFF" : "#0B1220"}>
            {node.live ? <AnimatedCount value={node.primary.reading.count} /> : ui.icon}
          </text>
          {node.test && (
            <g><rect x={x - r - 6} y={y - r - 4} width="44" height="18" rx="4" fill="#6D28D9" stroke="#0B1220" strokeWidth="1.5" />
              <text x={x - r + 16} y={y - r + 9} textAnchor="middle" fontSize="11" fontWeight="700" fill="#fff">TEST</text></g>
          )}
          {node.zones.length > 1 && (
            <g><circle cx={x + r - 2} cy={y - r + 2} r="11" fill="#7C3AED" stroke="#0B1220" strokeWidth="2" />
              <text x={x + r - 2} y={y - r + 6.5} textAnchor="middle" fontSize="12" fontWeight="700" fill="#fff">+{node.zones.length - 1}</text></g>
          )}
        </>
      ) : (
        <circle cx={x} cy={y} r={r - 8} fill="#0F192D" stroke="#3B5182" strokeWidth="2.5" strokeDasharray="5 5" />
      )}
      <text x={label.x} y={label.y1} textAnchor={label.anchor} fontSize="22" fontWeight="600" fill="#F1F5F9">{node.label}</text>
      <text x={label.x} y={label.y2} textAnchor={label.anchor} fontSize="17" fontWeight="700"
        fill={ui ? ui.fill : "#94A3B8"}>
        {hasCamera ? `${ui.icon} ${node.level}` : node.kind === "gate" ? (safestExit ? "SAFEST EXIT" : "exit gate") : "no camera"}
      </text>
    </g>
  );
}

/** The schematic map: live risk per camera node, walkways by live pressure, and the route to the
 * safest gate for the selected zone. Positions come from config/site_map.json (schematic). */
export default function GhatMap({ zones, now, connected, thresholds, selectedId, onSelect }) {
  const [siteMap, setSiteMap] = useState(null);
  const [error, setError] = useState(null);
  const [hovered, setHovered] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api.siteMap().then((value) => !cancelled && setSiteMap(value), (e) => !cancelled && setError(e.message));
    return () => { cancelled = true; };
  }, []);

  if (error) return <section className="rounded-xl bg-navy-800 p-4 text-sm text-red-200" role="alert">Map unavailable: {error}</section>;
  if (!siteMap) return <section className="grid h-64 place-items-center rounded-xl bg-navy-800 text-sm text-slate-400" aria-busy="true">Loading the site map…</section>;

  const model = buildMapModel({ siteMap, zones, now, connected, thresholds, selectedId });
  const { width, height } = siteMap;
  const hoveredNode = model.nodes.find((n) => n.id === hovered);
  const routeLine = model.route?.points.map(([x, y]) => `${x},${y}`).join(" ");

  return (
    <section aria-labelledby="map-heading" className="rounded-xl border border-navy-700 bg-navy-900 p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="map-heading" className="font-display text-sm font-semibold uppercase tracking-wider text-slate-300">Ramkund area · live risk map</h2>
        <p className="rounded bg-navy-800 px-2 py-0.5 text-xs font-semibold text-saffron">{siteMap.note}</p>
      </div>
      <div className="relative mt-2">
        <svg viewBox={`0 0 ${width} ${height}`} className="block h-auto max-h-[64vh] w-full select-none" role="group"
          aria-label={`${siteMap.note}. ${model.route ? `Safest exit for the selected zone: ${model.route.gate.replace("EXIT_GATE_", "Gate ")}.` : ""}`}>
          <defs>
            <pattern id="map-grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#14213D" strokeWidth="1" />
            </pattern>
            <filter id="route-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          <rect width={width} height={height} fill="url(#map-grid)" pointerEvents="none" />
          {model.edges.map((edge) => {
            const style = edge.band ? WALKWAY[edge.band] : null;
            return (
              <line key={`${edge.a}-${edge.b}`} x1={edge.from.x} y1={edge.from.y} x2={edge.to.x} y2={edge.to.y}
                stroke={style?.stroke ?? "#2A3D66"} strokeWidth={style?.width ?? 3} strokeLinecap="round"
                strokeDasharray={style ? undefined : "2 7"} opacity={style ? 0.9 : 0.8}>
                <title>{`Walkway ${edge.from.label} – ${edge.to.label}, ${edge.metres} m (schematic)${edge.pressure != null ? `, pressure ${formatNumber(edge.pressure)}` : ", no camera at either end"}`}</title>
              </line>
            );
          })}
          {routeLine && (
            <g aria-hidden="true">
              <polyline points={routeLine} fill="none" stroke="#F2A54A" strokeWidth="7" strokeLinejoin="round" strokeLinecap="round" opacity="0.35" filter="url(#route-glow)" />
              <polyline points={routeLine} fill="none" stroke="#F8C98F" strokeWidth="4" strokeLinejoin="round" strokeLinecap="round"
                strokeDasharray="12 12" className="motion-safe-only animate-route-dash" />
            </g>
          )}
          {model.nodes.map((node) => (
            <Node key={node.id} node={node} selected={model.selectedNode === node.id} safestExit={model.route?.gate === node.id}
              onSelect={onSelect} onHover={setHovered} />
          ))}
        </svg>
        {hoveredNode && <Tooltip node={hoveredNode} now={now} connected={connected} width={width} height={height} />}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-300" aria-label="Map key">
        {["GREEN", "YELLOW", "ORANGE", "RED"].map((level) => (
          <li key={level} className="flex items-center gap-1.5">
            <span aria-hidden="true" className="inline-block h-3 w-3 rounded-full" style={{ background: levelUi(level).fill }} />
            {levelUi(level).icon} {level}
          </li>
        ))}
        <li className="flex items-center gap-1.5"><span aria-hidden="true" className="inline-block h-0.5 w-6 bg-slate-500" />walkway (thicker and warmer = more crowd pressure)</li>
        <li className="flex items-center gap-1.5"><span aria-hidden="true" className="inline-block h-1 w-6 rounded bg-saffron" />route to the safest gate</li>
        <li>dashed circle = no camera</li>
      </ul>
      {model.route?.throughTest && (
        <p className="mt-1 text-xs text-violet-200">
          This route passes a test-video camera (marked TEST): test videos never change the live exit advice, so the route ignores their crowd.
        </p>
      )}
      {model.offMap.length > 0 && (
        <p className="mt-1 text-xs text-slate-400">Not on the map (no exit-map location): {model.offMap.map((z) => z.name).join(", ")}.</p>
      )}
    </section>
  );
}
