import { useEffect, useState } from "react";
import { videoApi } from "../api.js";
import { formatTime } from "../format.js";
import { routeHref } from "../useHashRoute.js";
import ModelNotice from "./ModelNotice.jsx";
import VideoUploadForm, { ProgressBar } from "./VideoUploadForm.jsx";

const STATE_TEXT = { queued: "Waiting in the queue", running: "Analysing", done: "Done", failed: "Failed" };

function JobRow({ job, onDelete }) {
  return (
    <li className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 py-3 first:border-t-0">
      <div className="min-w-0 flex-1 basis-64">
        <a href={routeHref.report(job.id)} className="font-bold text-sky-900 underline underline-offset-2">
          {job.name}
        </a>
        <p className="text-sm text-slate-700">
          {job.video_name} · {job.area_m2} m² · started {formatTime(job.created_at * 1000)}
        </p>
        {job.state === "running" ? (
          <div className="mt-1 max-w-sm"><ProgressBar label="Analysing" percent={job.progress} /></div>
        ) : (
          <p className={`text-sm font-semibold ${job.state === "failed" ? "text-red-800" : "text-slate-800"}`}>
            {STATE_TEXT[job.state]}{job.state === "failed" && job.error ? `: ${job.error}` : ""}
          </p>
        )}
      </div>
      <button type="button" onClick={() => onDelete(job)}
        className="rounded-md border-2 border-slate-900 px-3 py-1.5 text-sm font-bold text-slate-900 hover:bg-slate-100 focus:outline-none focus-visible:ring-4 focus-visible:ring-sky-500">
        Delete
      </button>
    </li>
  );
}

/** Upload a video, then analyse it once or loop it as a live zone; past analyses and video zones below. */
export default function VideoPage({ jobs, onJob, videoZones }) {
  const [options, setOptions] = useState(null);
  const [optionsError, setOptionsError] = useState(null);
  const [message, setMessage] = useState(null);
  const [actionError, setActionError] = useState(null);

  useEffect(() => {
    videoApi.options().then(setOptions, (error) => setOptionsError(error.message));
  }, []);

  async function analyse(video, body) {
    const job = await videoApi.analyse(video.id, body);
    onJob(job);
    window.location.hash = routeHref.report(job.id);
  }

  async function addLive(video, body) {
    const zone = await videoApi.addLive(video.id, body);
    setMessage(`"${zone.name}" now runs as live zone ${zone.id}, looping the video. It is on the live dashboard.`);
  }

  async function act(action) {
    setActionError(null);
    try {
      await action();
    } catch (error) {
      setActionError(error.message);
    }
  }

  const deleteJob = (job) =>
    window.confirm(`Delete the analysis "${job.name}" and its files?`) && act(() => videoApi.deleteJob(job.id).then(() => onJob({ ...job, state: "deleted" })));
  const removeZone = (zone) => act(() => videoApi.removeZone(zone.id));

  return (
    <main className="mx-auto max-w-5xl space-y-6 px-3 py-4 sm:px-6">
      <div>
        <h2 className="text-2xl font-bold">Video analysis</h2>
        <p className="mt-1 text-slate-800">
          Run a recorded crowd video through exactly the live cameras' pipeline: one 5-frame burst every 2 s of
          video, CSRNet count, optical-flow turbulence, crowd pressure and the 30 s risk level.
        </p>
      </div>

      {optionsError && (
        <p role="alert" className="rounded-md border-l-8 border-red-700 bg-red-50 p-3 font-semibold text-red-950">
          Cannot reach the backend: {optionsError}
        </p>
      )}
      {options && !options.model_loaded && (
        <p role="alert" className="rounded-md border-l-8 border-red-700 bg-red-50 p-3 font-semibold text-red-950">
          No trained model: counts unavailable, so videos cannot be analysed. {options.model_error}
        </p>
      )}
      <ModelNotice model={options?.model} />
      {message && (
        <p role="status" className="rounded-md border-l-8 border-green-700 bg-green-50 p-3 font-semibold text-green-950">
          {message} <a href={routeHref.live} className="underline">Open the live dashboard</a>
        </p>
      )}

      <VideoUploadForm options={options} onAnalyse={analyse} onAddLive={addLive} />

      {actionError && (
        <p role="alert" className="rounded-md border-l-8 border-red-700 bg-red-50 p-3 font-semibold text-red-950">{actionError}</p>
      )}

      <section aria-labelledby="video-zones-heading" className="rounded-xl bg-white p-4 shadow-sm sm:p-6">
        <h3 id="video-zones-heading" className="text-lg font-bold">Videos running as live cameras</h3>
        {videoZones.length === 0 ? (
          <p className="mt-2 text-sm text-slate-700">None. Use "Add as live camera" after uploading.</p>
        ) : (
          <ul className="mt-2">
            {videoZones.map((zone) => (
              <li key={zone.id} className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 py-3 first:border-t-0">
                <span><span className="font-bold">{zone.name}</span> <span className="text-sm text-slate-700">({zone.id})</span></span>
                <button type="button" onClick={() => removeZone(zone)}
                  className="rounded-md bg-red-800 px-3 py-1.5 text-sm font-bold text-white focus:outline-none focus-visible:ring-4 focus-visible:ring-sky-500">
                  Remove from live dashboard
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section aria-labelledby="jobs-heading" className="rounded-xl bg-white p-4 shadow-sm sm:p-6">
        <h3 id="jobs-heading" className="text-lg font-bold">Analyses</h3>
        <p className="text-sm text-slate-700">They run one at a time, and slow down if the live cameras need the time.</p>
        {jobs.length === 0 ? (
          <p className="mt-2 text-sm text-slate-700">No analyses yet.</p>
        ) : (
          <ul className="mt-2">{jobs.map((job) => <JobRow key={job.id} job={job} onDelete={deleteJob} />)}</ul>
        )}
      </section>
    </main>
  );
}
