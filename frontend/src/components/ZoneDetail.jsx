import { useEffect, useRef, useState } from "react";
import { countMethodLabel, formatCount, formatTime } from "../format.js";
import DensityLegend from "./DensityLegend.jsx";
import GridOverlay, { GridCaption } from "./GridOverlay.jsx";
import PressureChart from "./PressureChart.jsx";
import RiskBadge from "./RiskBadge.jsx";

export const OVERLAYS = [
  { value: "none", label: "Camera only" },
  { value: "density", label: "Density heatmap" },
  { value: "detections", label: "Detections" },
  { value: "gradcam", label: "Grad-CAM" },
];

const OVERLAY_HELP = {
  density: "Where CSRNet counts people, coloured by persons per m² on a fixed scale (key below).",
  detections: "People found by YOLOv8 (one box each). It misses small, distant or hidden people, so in a dense crowd the count comes from CSRNet.",
  gradcam: "Which image regions drove the predicted count (Grad-CAM).",
};

/** Zone details: live snapshot with an optional overlay and the pressure trend. */
export default function ZoneDetail({ zone, level, feed, overlay, onOverlayChange, thresholds, densityScale, onClose }) {
  const closeButton = useRef(null);
  const [showGrid, setShowGrid] = useState(true);
  const reading = zone.reading;

  useEffect(() => {
    closeButton.current?.focus();
    const onKey = (event) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-30 overflow-y-auto bg-slate-900/70 sm:p-6" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="zone-detail-title"
        className="mx-auto min-h-full w-full max-w-4xl bg-white p-4 shadow-xl sm:min-h-0 sm:rounded-xl sm:p-6"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <h2 id="zone-detail-title" className="text-xl font-bold sm:text-2xl">{zone.name}</h2>
            <RiskBadge level={level} />
          </div>
          <button ref={closeButton} type="button" onClick={onClose}
            className="rounded-md bg-slate-900 px-4 py-2 font-bold text-white focus:outline-none focus-visible:ring-4 focus-visible:ring-sky-500">
            Close
          </button>
        </div>

        <fieldset className="mt-4">
          <legend className="text-sm font-semibold text-slate-800">Overlay</legend>
          <div className="mt-1 inline-flex flex-wrap overflow-hidden rounded-lg ring-2 ring-slate-900">
            {OVERLAYS.map((option) => (
              <label key={option.value}
                className={`cursor-pointer px-3 py-2 text-sm font-bold has-[:focus-visible]:ring-4 has-[:focus-visible]:ring-sky-500 ${
                  overlay === option.value ? "bg-slate-900 text-white" : "bg-white text-slate-900"
                }`}>
                <input type="radio" name="overlay" value={option.value} checked={overlay === option.value}
                  onChange={() => onOverlayChange(option.value)} className="sr-only" />
                {overlay === option.value && <span aria-hidden="true">✓ </span>}
                {option.label}
              </label>
            ))}
          </div>
          {OVERLAY_HELP[overlay] && <p className="mt-1 text-sm text-slate-700">{OVERLAY_HELP[overlay]}</p>}
          {overlay === "density" && <DensityLegend scale={densityScale} />}
          <label className="mt-2 flex items-center gap-2 text-sm font-semibold text-slate-800">
            <input type="checkbox" checked={showGrid} onChange={(event) => setShowGrid(event.target.checked)}
              className="h-4 w-4" />
            Show grid (people per cell, hotspot outlined)
          </label>
        </fieldset>

        <figure className="mt-3">
          <div className="relative overflow-hidden rounded-lg bg-slate-800">
            {feed.snapshot ? (
              <>
                <img src={feed.snapshot} alt={`Camera view of ${zone.name}`} className="block h-auto w-full" />
                {overlay !== "none" && feed.overlay && (
                  <img src={feed.overlay} alt={`${OVERLAYS.find((o) => o.value === overlay).label} overlay`}
                    className="absolute inset-0 h-full w-full" />
                )}
                {showGrid && <GridOverlay grid={reading?.grid} />}
              </>
            ) : (
              <p className="p-8 text-center font-semibold text-white">
                {feed.snapshotError ? `Camera image unavailable: ${feed.snapshotError}` : "Loading camera image…"}
              </p>
            )}
          </div>
          <figcaption className="mt-1 text-sm text-slate-700">
            {feed.capturedAt ? `Frame captured ${formatTime(feed.capturedAt * 1000)}, refreshes every 2 s.` : ""}
            {reading?.count_method && (
              <span className="block">
                Count: {formatCount(reading.count)} people by {countMethodLabel(reading.count_method)}
                {reading.detection_count != null && ` (YOLO found ${formatCount(reading.detection_count)}, CSRNet counted ${formatCount(reading.density_count)})`}.
              </span>
            )}
            {showGrid && <GridCaption grid={reading?.grid} />}
            {overlay !== "none" && feed.overlayError && (
              <span className="block font-semibold text-red-800">Overlay unavailable: {feed.overlayError}</span>
            )}
          </figcaption>
        </figure>

        <div className="mt-6">
          <PressureChart points={zone.history} thresholds={thresholds} />
        </div>
      </div>
    </div>
  );
}
