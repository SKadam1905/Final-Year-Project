import { useEffect, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { jobUrl, sampleUrl, videoApi } from "../api.js";
import { countMethodLabel, formatCount, formatDuration, formatNumber, prettyNode } from "../format.js";
import { riskStyle } from "../risk.js";
import { routeHref } from "../useHashRoute.js";
import DensityLegend from "./DensityLegend.jsx";
import GridOverlay, { GridCaption } from "./GridOverlay.jsx";
import ModelNotice from "./ModelNotice.jsx";
import ReportSummary from "./ReportSummary.jsx";
import RiskBadge from "./RiskBadge.jsx";
import { ProgressBar } from "./VideoUploadForm.jsx";

const THRESHOLD_LINES = [
  { key: "yellow", name: "Yellow", stroke: "#a16207" },
  { key: "orange", name: "Orange", stroke: "#c2410c" },
  { key: "red", name: "Red", stroke: "#b91c1c" },
];
const OVERLAYS = [
  { value: "density", label: "Density heatmap" },
  { value: "detections", label: "Detections" },
  { value: "gradcam", label: "Grad-CAM" },
  { value: "none", label: "Frame only" },
];
const AXIS_TICK = { fill: "#1e293b", fontSize: 12 };

/** Time ranges of consecutive "moving camera" samples, each sample covering +-1 s, for shading. */
export function movingCameraRanges(samples, halfWidth = 1) {
  const ranges = [];
  for (const s of samples) {
    if (!s.moving_camera) continue;
    const last = ranges.at(-1);
    if (last && s.time_s - halfWidth <= last.to + 0.01) last.to = s.time_s + halfWidth;
    else ranges.push({ from: s.time_s - halfWidth, to: s.time_s + halfWidth });
  }
  return ranges;
}

const formatValue = (value) => (value == null ? "—" : Number(value).toFixed(2));

/** Shading for moving-camera samples and dashed lines at scene cuts (reused by both charts). */
function SampleMarkers({ samples, yAxisId }) {
  const axis = yAxisId ? { yAxisId } : {};
  return (
    <>
      {movingCameraRanges(samples).map((r) => (
        <ReferenceArea key={`cam-${r.from}`} x1={r.from} x2={r.to} fill="#7c3aed" fillOpacity={0.12} strokeOpacity={0}
          ifOverflow="hidden" {...axis} />
      ))}
      {samples.filter((s) => s.scene_cut).map((s) => (
        <ReferenceLine key={`cut-${s.index}`} x={s.time_s} stroke="#64748b" strokeDasharray="2 3" {...axis} />
      ))}
    </>
  );
}

function MarkerKey({ samples }) {
  const cuts = samples.filter((s) => s.scene_cut).length;
  const moving = samples.filter((s) => s.moving_camera).length;
  if (!cuts && !moving) return null;
  return (
    <p className="mt-1 text-xs text-slate-700">
      {moving > 0 && <><span className="mr-1 inline-block h-3 w-4 align-middle bg-violet-600/20 ring-1 ring-violet-600/40" />
        Moving camera ({moving}): its motion was removed before turbulence. </>}
      {cuts > 0 && <><span className="mx-1 inline-block w-4 align-middle border-t-2 border-dashed border-slate-500" />
        Scene cut ({cuts}): skipped, a gap in the lines.</>}
    </p>
  );
}

/** Recharts passes the clicked point's index as a number or a numeric string. */
function clickedIndex(state) {
  const index = Number(state?.activeTooltipIndex ?? state?.activeIndex);
  return Number.isInteger(index) && index >= 0 ? index : null;
}

function TimelineCharts({ samples, thresholds, selected, onSelect }) {
  const data = samples.map((s) => ({ ...s, t: s.time_s }));
  const top = Math.ceil(Math.max(thresholds.red * 1.25, ...samples.map((s) => s.raw_pressure), 1));
  const select = (state) => {
    const index = clickedIndex(state);
    if (index !== null) onSelect(index);
  };
  const selectedLine = samples[selected] && (
    <ReferenceLine x={samples[selected].time_s} stroke="#0369a1" strokeWidth={2} />
  );
  return (
    <div className="space-y-6">
      <figure>
        <figcaption className="mb-1 text-sm font-semibold text-slate-800">
          Crowd pressure over video time (30 s mean, as the live risk level uses). Click a point to see that frame.
        </figcaption>
        <div className="h-64 w-full cursor-pointer">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} onClick={select} margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
              <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={formatDuration}
                tick={AXIS_TICK} minTickGap={30} />
              <YAxis domain={[0, top]} allowDecimals={false} tick={AXIS_TICK} />
              <Tooltip labelFormatter={(t) => `t = ${formatDuration(t)}`}
                formatter={(value, name) => [formatValue(value), name]} />
              <Legend />
              <SampleMarkers samples={samples} />
              {THRESHOLD_LINES.map((line) => (
                <ReferenceLine key={line.key} y={thresholds[line.key]} stroke={line.stroke} strokeWidth={2}
                  strokeDasharray="6 4" ifOverflow="extendDomain"
                  label={{ value: `${line.name} ${thresholds[line.key]}`, position: "insideTopLeft", fill: line.stroke,
                    fontSize: 12, fontWeight: 700 }} />
              ))}
              {selectedLine}
              <Line name="Pressure (30 s mean)" dataKey="pressure" stroke="#0f172a" strokeWidth={2.5} dot={{ r: 2 }}
                activeDot={{ r: 6 }} isAnimationActive={false} />
              <Line name="Pressure (this sample)" dataKey="raw_pressure" stroke="#64748b" strokeWidth={1}
                strokeDasharray="3 3" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <MarkerKey samples={samples} />
      </figure>
      <figure>
        <figcaption className="mb-1 text-sm font-semibold text-slate-800">
          People counted and turbulence (after removing camera motion)
        </figcaption>
        <div className="h-56 w-full cursor-pointer">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} onClick={select} margin={{ top: 8, right: 0, bottom: 4, left: -8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
              <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={formatDuration}
                tick={AXIS_TICK} minTickGap={30} />
              <YAxis yAxisId="count" tick={AXIS_TICK} allowDecimals={false} />
              <YAxis yAxisId="turbulence" orientation="right" domain={[0, 1]} tick={AXIS_TICK} />
              <Tooltip labelFormatter={(t) => `t = ${formatDuration(t)}`}
                formatter={(value, name) => [value == null ? "—" : name === "People" ? formatCount(value) : formatValue(value), name]} />
              <Legend />
              <SampleMarkers samples={samples} yAxisId="count" />
              {selectedLine && <ReferenceLine yAxisId="count" x={samples[selected].time_s} stroke="#0369a1" strokeWidth={2} />}
              <Line yAxisId="count" name="People" dataKey="count" stroke="#1d4ed8" strokeWidth={2} dot={false}
                activeDot={{ r: 6 }} isAnimationActive={false} />
              <Line yAxisId="turbulence" name="Turbulence (0-1)" dataKey="turbulence" stroke="#b45309" strokeWidth={2}
                dot={false} activeDot={{ r: 6 }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </figure>
    </div>
  );
}

function FrameViewer({ jobId, samples, selected, onSelect, densityScale }) {
  const [overlay, setOverlay] = useState("density");
  const [overlayError, setOverlayError] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const sample = samples[selected];
  useEffect(() => setOverlayError(false), [selected, overlay]);
  if (!sample) return null;
  return (
    <section aria-labelledby="frame-heading">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 id="frame-heading" className="text-lg font-bold">
          Frame at {formatDuration(sample.time_s)} <span className="font-normal text-slate-700">(sample {selected + 1} of {samples.length})</span>
        </h3>
        {sample.scene_cut ? (
          <span className="rounded bg-slate-700 px-2 py-0.5 text-sm font-bold text-white">Scene cut: skipped</span>
        ) : (
          <RiskBadge level={riskStyle(sample.risk_level)} size="sm" />
        )}
      </div>
      {sample.moving_camera && (
        <p className="mt-1 inline-block rounded bg-violet-800 px-2 py-0.5 text-sm font-bold text-white">
          Moving camera: {formatNumber(sample.camera_motion)} px/frame of camera motion removed before turbulence
        </p>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-3">
        <button type="button" onClick={() => onSelect(Math.max(0, selected - 1))} disabled={selected === 0}
          className="rounded-md border-2 border-slate-900 px-3 py-1 text-sm font-bold disabled:opacity-40">◀ Previous</button>
        <label className="flex flex-1 items-center gap-2 text-sm font-semibold">
          <span className="sr-only">Sample</span>
          <input type="range" min={0} max={samples.length - 1} value={selected} onChange={(e) => onSelect(Number(e.target.value))}
            className="w-full" aria-label="Sample in the video" />
        </label>
        <button type="button" onClick={() => onSelect(Math.min(samples.length - 1, selected + 1))}
          disabled={selected === samples.length - 1}
          className="rounded-md border-2 border-slate-900 px-3 py-1 text-sm font-bold disabled:opacity-40">Next ▶</button>
      </div>
      <fieldset className="mt-3">
        <legend className="text-sm font-semibold text-slate-800">Overlay</legend>
        <div className="mt-1 inline-flex flex-wrap overflow-hidden rounded-lg ring-2 ring-slate-900">
          {OVERLAYS.map((option) => (
            <label key={option.value}
              className={`cursor-pointer px-3 py-2 text-sm font-bold has-[:focus-visible]:ring-4 has-[:focus-visible]:ring-sky-500 ${
                overlay === option.value ? "bg-slate-900 text-white" : "bg-white text-slate-900"}`}>
              <input type="radio" name="report-overlay" value={option.value} checked={overlay === option.value}
                onChange={() => setOverlay(option.value)} className="sr-only" />
              {overlay === option.value && <span aria-hidden="true">✓ </span>}
              {option.label}
            </label>
          ))}
        </div>
        <label className="mt-2 flex items-center gap-2 text-sm font-semibold text-slate-800">
          <input type="checkbox" checked={showGrid} onChange={(event) => setShowGrid(event.target.checked)} className="h-4 w-4" />
          Show grid (people per cell, hotspot outlined)
        </label>
      </fieldset>
      <figure className="mt-2">
        <div className="relative overflow-hidden rounded-lg bg-slate-800">
          <img src={sampleUrl(jobId, selected, "frame.jpg")} alt={`Video frame at ${formatDuration(sample.time_s)}`}
            className="block h-auto w-full" />
          {overlay !== "none" && !overlayError && !sample.scene_cut && (
            <img src={sampleUrl(jobId, selected, `${overlay}.png`)} alt={`${OVERLAYS.find((o) => o.value === overlay).label} overlay`}
              onError={() => setOverlayError(true)} className="absolute inset-0 h-full w-full" />
          )}
          {showGrid && !sample.scene_cut && <GridOverlay grid={sample.grid} />}
        </div>
        <figcaption className="mt-1 text-sm text-slate-800">
          {sample.scene_cut
            ? "The camera switched shots inside this burst, so it was not analysed and is left out of the risk average and the summary."
            : <>{formatCount(sample.count)} people · {formatNumber(sample.density)} /m² · turbulence{" "}
              {formatNumber(sample.turbulence)} · pressure {formatNumber(sample.pressure)} (30 s mean)
              {sample.count_method && <span className="block">Counted by {countMethodLabel(sample.count_method)}
                {sample.detection_count != null && ` (YOLO found ${formatCount(sample.detection_count)}, CSRNet counted ${formatCount(sample.density_count)})`}.</span>}
              {showGrid && <GridCaption grid={sample.grid} />}</>}
          {overlayError && <span className="block font-semibold text-red-800">
            {overlay === "detections" ? "No person detections for this sample (analysed without the detector)." : "The overlay could not be loaded."}</span>}
        </figcaption>
        {overlay === "density" && !sample.scene_cut && <DensityLegend scale={densityScale} />}
      </figure>
    </section>
  );
}

function ExitsAtPeak({ peak }) {
  return (
    <section aria-labelledby="exits-heading">
      <h3 id="exits-heading" className="text-lg font-bold">Recommended exits at the peak ({formatDuration(peak.time_s)})</h3>
      {!peak.graph_node ? (
        <p className="mt-1 text-sm text-slate-700">No exit-map location was chosen for this video, so no exits were worked out.</p>
      ) : (
        <>
          <p className="mt-1 text-sm text-slate-700">
            From {prettyNode(peak.graph_node)} at pressure {formatNumber(peak.pressure)}, with the other areas assumed clear.
          </p>
          <ol className="mt-1 space-y-1 text-sm">
            {peak.exits.map((e, i) => (
              <li key={e.exit}>
                <span className="font-bold">{i + 1}. {prettyNode(e.exit)}</span>
                <span className="text-slate-700"> — {e.path.map(prettyNode).join(" → ")}</span>
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}

function AlertsList({ alerts, telegram }) {
  return (
    <section aria-labelledby="video-alerts-heading">
      <h3 id="video-alerts-heading" className="text-lg font-bold">Alerts</h3>
      {alerts.length === 0 ? (
        <p className="mt-1 text-sm text-slate-700">The risk level never reached ORANGE, so no alert would have been sent.</p>
      ) : (
        <>
          <p className="mt-1 text-sm text-slate-700">
            {telegram ? "Sent to Telegram, marked [VIDEO TEST]." : "Not sent anywhere: Telegram alerts were off for this video."}
          </p>
          <ol className="mt-2 space-y-2 text-sm">
            {alerts.map((a) => (
              <li key={`${a.sample}-${a.risk_level}`} className="flex flex-wrap items-center gap-2">
                <RiskBadge level={riskStyle(a.risk_level)} size="sm" />
                <span className="font-semibold tabular-nums">{formatDuration(a.time_s)}</span>
                <span className="min-w-0 flex-1 basis-64">{a.message}</span>
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}

function Downloads({ job, onJob }) {
  const [error, setError] = useState(null);
  const annotated = job.annotated ?? { state: "none" };
  async function request() {
    setError(null);
    try {
      onJob(await videoApi.requestAnnotated(job.id));
    } catch (e) {
      setError(e.message);
    }
  }
  const link = "inline-block rounded-md bg-slate-900 px-4 py-2 font-bold text-white focus:outline-none focus-visible:ring-4 focus-visible:ring-sky-500";
  return (
    <section aria-labelledby="downloads-heading">
      <h3 id="downloads-heading" className="text-lg font-bold">Downloads</h3>
      <div className="mt-2 flex flex-wrap items-center gap-3">
        <a href={jobUrl(job.id, "results.csv")} download className={link}>Results CSV</a>
        {annotated.state === "done" ? (
          <a href={jobUrl(job.id, "annotated.mp4")} download className={link}>Annotated MP4</a>
        ) : ["queued", "running"].includes(annotated.state) ? (
          <div className="w-64"><ProgressBar label="Making annotated video" percent={annotated.progress ?? 0} /></div>
        ) : (
          <button type="button" onClick={request} className={`${link} bg-violet-800`}>
            Make annotated MP4 (slow)
          </button>
        )}
      </div>
      {annotated.state === "failed" && <p className="mt-1 text-sm font-semibold text-red-800">Annotated video failed: {annotated.error}</p>}
      {error && <p role="alert" className="mt-1 text-sm font-semibold text-red-800">{error}</p>}
      <p className="mt-1 text-sm text-slate-700">The annotated MP4 shows the density heatmap, count and risk level on every frame.</p>
    </section>
  );
}

/** One analysis: progress while it runs, then the full report. `job` comes from useVideoJobs. */
export default function VideoReport({ jobId, job, loaded, onJob }) {
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(0);
  const done = job?.state === "done";

  useEffect(() => {
    if (!done) return undefined;
    let cancelled = false;
    videoApi.results(jobId).then(
      (body) => {
        if (cancelled) return;
        setResults(body);
        setSelected(body.summary.peak_pressure_sample ?? 0);
      },
      (e) => !cancelled && setError(e.message),
    );
    return () => {
      cancelled = true;
    };
  }, [jobId, done]);

  async function remove() {
    if (!window.confirm(`Delete the analysis "${job.name}" and its files?`)) return;
    try {
      await videoApi.deleteJob(job.id);
      onJob({ ...job, state: "deleted" });
      window.location.hash = routeHref.video;
    } catch (e) {
      setError(e.message);
    }
  }

  if (!job) {
    return (
      <main className="mx-auto max-w-5xl px-3 py-6 sm:px-6">
        <p className="font-semibold">{loaded ? "This analysis does not exist (it may have been deleted)." : "Loading…"}</p>
        <a href={routeHref.video} className="mt-2 inline-block font-bold text-sky-900 underline">Back to Video analysis</a>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-5xl space-y-6 px-3 py-4 sm:px-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <a href={routeHref.video} className="text-sm font-bold text-sky-900 underline">← Video analysis</a>
          <h2 className="mt-1 text-2xl font-bold">{job.name}</h2>
          <p className="text-slate-800">{job.video_name} · visible area {job.area_m2} m²</p>
        </div>
        <button type="button" onClick={remove}
          className="rounded-md border-2 border-red-800 px-3 py-1.5 text-sm font-bold text-red-800 hover:bg-red-50 focus:outline-none focus-visible:ring-4 focus-visible:ring-sky-500">
          Delete analysis
        </button>
      </div>

      {job.state === "queued" && (
        <p role="status" className="rounded-md bg-white p-4 font-semibold shadow-sm">
          Waiting in the queue: analyses run one at a time.
        </p>
      )}
      {job.state === "running" && (
        <div role="status" className="rounded-md bg-white p-4 shadow-sm">
          <ProgressBar label="Analysing the video" percent={job.progress} />
          {job.slowed_for_live > 0 && (
            <p className="mt-2 text-sm text-slate-700">Paused {job.slowed_for_live} times so the live cameras stayed on time.</p>
          )}
        </div>
      )}
      {job.state === "failed" && (
        <p role="alert" className="rounded-md border-l-8 border-red-700 bg-red-50 p-3 font-semibold text-red-950">
          The analysis failed: {job.error}
        </p>
      )}
      {error && <p role="alert" className="rounded-md border-l-8 border-red-700 bg-red-50 p-3 font-semibold text-red-950">{error}</p>}

      {done && results && (
        <>
          <ModelNotice model={results.model} />
          <div className="rounded-xl bg-white p-4 shadow-sm sm:p-6"><ReportSummary summary={results.summary} /></div>
          <div className="rounded-xl bg-white p-4 shadow-sm sm:p-6">
            <TimelineCharts samples={results.samples} thresholds={results.settings.thresholds} selected={selected}
              onSelect={setSelected} />
          </div>
          <div className="rounded-xl bg-white p-4 shadow-sm sm:p-6">
            <FrameViewer jobId={job.id} samples={results.samples} selected={selected} onSelect={setSelected}
              densityScale={results.settings.density_scale} />
          </div>
          <div className="grid gap-6 rounded-xl bg-white p-4 shadow-sm sm:p-6 lg:grid-cols-2">
            <ExitsAtPeak peak={results.exits_at_peak} />
            <AlertsList alerts={results.alerts} telegram={job.telegram_alerts} />
          </div>
          <div className="rounded-xl bg-white p-4 shadow-sm sm:p-6"><Downloads job={job} onJob={onJob} /></div>
        </>
      )}
    </main>
  );
}
