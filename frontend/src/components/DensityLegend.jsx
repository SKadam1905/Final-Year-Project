/**
 * Colour key for the density heatmap: one fixed persons/m² scale (from /api/config or a report's
 * settings), the same for every frame and camera. `scale` = { unit, max, stops: [{ value, colour }] }.
 */
export default function DensityLegend({ scale, tone = "light" }) {
  const dark = tone === "dark";
  if (!scale?.stops?.length) return null;
  const gradient = scale.stops.map((s) => `${s.colour} ${(100 * s.value) / scale.max}%`).join(", ");
  return (
    <figure className="mt-2 max-w-sm" aria-label={`Heatmap colour key in ${scale.unit}`}>
      <div className="h-3 rounded-sm ring-1 ring-slate-400" style={{ background: `linear-gradient(to right, ${gradient})` }} />
      <div className={`relative mt-0.5 h-4 text-xs font-semibold tabular-nums ${dark ? "text-slate-200" : "text-slate-800"}`}>
        {scale.stops.map((s, i) => {
          const last = i === scale.stops.length - 1;
          const position = i === 0 ? { left: 0 } : last ? { right: 0 }
            : { left: `${(100 * s.value) / scale.max}%`, transform: "translateX(-50%)" };
          return (
            <span key={s.value} className="absolute" style={position}>{last ? `${s.value}+` : s.value}</span>
          );
        })}
      </div>
      <figcaption className={`text-xs ${dark ? "text-slate-300" : "text-slate-700"}`}>
        Density in {scale.unit}, the same scale for every frame and camera: green is sparse, yellow from{" "}
        {scale.stops[1]?.value}, orange from {scale.stops[2]?.value}, red from {scale.stops[3]?.value}.
      </figcaption>
    </figure>
  );
}
