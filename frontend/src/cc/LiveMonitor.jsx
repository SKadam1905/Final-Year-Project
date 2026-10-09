import { useState } from "react";
import { formatTime } from "../format.js";
import { useZoneFeed } from "../useLiveZones.js";
import DensityLegend from "../components/DensityLegend.jsx";
import GridOverlay, { GridCaption } from "../components/GridOverlay.jsx";

const OVERLAYS = [
  { value: "none", label: "None" },
  { value: "detections", label: "Detections" },
  { value: "density", label: "Density heatmap" },
  { value: "grid", label: "Grid + hotspot" },
];

/** The selected camera, refreshed every 2 s, with one overlay at a time and its key. */
export default function LiveMonitor({ zone, densityScale }) {
  const [overlay, setOverlay] = useState("none");
  const feed = useZoneFeed(zone?.id ?? null, overlay === "grid" || overlay === "none" ? "none" : overlay);
  if (!zone) return null;
  const label = OVERLAYS.find((o) => o.value === overlay).label;

  return (
    <section aria-labelledby="monitor-heading" className="rounded-xl border border-navy-700 bg-navy-800 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 id="monitor-heading" className="font-display text-sm font-semibold uppercase tracking-wider text-slate-300">Live monitoring</h3>
        <fieldset>
          <legend className="sr-only">Overlay</legend>
          <div className="flex flex-wrap overflow-hidden rounded-lg ring-1 ring-navy-500">
            {OVERLAYS.map((option) => (
              <label key={option.value}
                className={`cursor-pointer px-2 py-1 text-xs font-semibold has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-saffron ${
                  overlay === option.value ? "bg-saffron text-navy-950" : "bg-navy-900 text-slate-300 hover:text-white"}`}>
                <input type="radio" name="cc-overlay" value={option.value} checked={overlay === option.value}
                  onChange={() => setOverlay(option.value)} className="sr-only" />
                {option.label}
              </label>
            ))}
          </div>
        </fieldset>
      </div>
      <figure className="mt-2">
        <div className="relative overflow-hidden rounded-lg bg-navy-950">
          {feed.snapshot ? (
            <>
              <img src={feed.snapshot} alt={`Camera view of ${zone.name}`} className="block h-auto w-full" />
              {overlay !== "none" && overlay !== "grid" && feed.overlay && (
                <img src={feed.overlay} alt={`${label} overlay`} className="absolute inset-0 h-full w-full" />
              )}
              {overlay === "grid" && <GridOverlay grid={zone.reading?.grid} />}
            </>
          ) : (
            <p className="grid aspect-video place-items-center p-4 text-center text-sm font-medium text-slate-300">
              {feed.snapshotError ? `Camera image unavailable: ${feed.snapshotError}` : "Loading camera image…"}
            </p>
          )}
        </div>
        <figcaption className="mt-1 space-y-1 text-xs text-slate-300">
          <span className="block">{feed.capturedAt ? `Frame captured ${formatTime(feed.capturedAt * 1000)} · refreshes every 2 s` : ""}</span>
          {overlay === "detections" && <span className="block"><span className="mr-1 inline-block h-3 w-3 border-2 border-cyan-400 align-middle" />a person found by YOLOv8 (misses small, distant or hidden people)</span>}
          {overlay === "grid" && <GridCaption grid={zone.reading?.grid} />}
          {feed.overlayError && overlay !== "grid" && overlay !== "none" && (
            <span className="block font-semibold text-red-300">Overlay unavailable: {feed.overlayError}</span>
          )}
        </figcaption>
        {overlay === "density" && <DensityLegend scale={densityScale} tone="dark" />}
      </figure>
    </section>
  );
}
