import { formatCount, hotspotText } from "../format.js";

/**
 * Zone-wise segmentation drawn over a camera image: people in each grid cell, the most crowded
 * cell (the hotspot) outlined in red. `grid` is the backend's { rows, cols, counts, hotspot }.
 * The drawing is decorative for screen readers; the caption below the image carries its text.
 */
export default function GridOverlay({ grid }) {
  if (!grid?.counts) return null;
  const { rows, cols, counts, hotspot } = grid;
  return (
    <div aria-hidden="true" data-testid="grid-overlay" className="pointer-events-none absolute inset-0 grid"
      style={{ gridTemplateRows: `repeat(${rows}, 1fr)`, gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
      {counts.flatMap((row, r) => row.map((count, c) => {
        const hot = hotspot?.row === r && hotspot?.col === c;
        return (
          <div key={`${r}-${c}`} className={hot ? "relative border-4 border-red-600 bg-red-600/15" : "relative border border-white/70"}>
            <span className={`absolute left-1 top-1 rounded px-1.5 py-0.5 text-xs font-bold tabular-nums text-white ${
              hot ? "bg-red-700" : "bg-slate-900/75"}`}>
              {hot ? `Hotspot · ${formatCount(count)}` : formatCount(count)}
            </span>
          </div>
        );
      }))}
    </div>
  );
}

/** One line for under the image: what the grid shows and where the hotspot is. */
export function GridCaption({ grid }) {
  if (!grid?.counts) return null;
  const spot = hotspotText(grid.hotspot);
  return (
    <span className="block">
      Grid: people in each of {grid.rows} × {grid.cols} cells. {spot ? `Hotspot: ${spot}.` : "No people counted."}
    </span>
  );
}
