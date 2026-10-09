import { useState } from "react";
import { uploadVideo } from "../api.js";
import { formatDuration, prettyNode } from "../format.js";
import { emptyVideoForm, fileError, nameFromFile, VIDEO_TYPES, zoneErrors, zoneRequest, areaFrom } from "../videoForm.js";

const inputClass =
  "mt-1 block w-full rounded-md border-2 border-slate-400 bg-white px-3 py-2 text-slate-900 focus:border-sky-600 focus:outline-none focus:ring-2 focus:ring-sky-600";

function FieldError({ id, message }) {
  if (!message) return null;
  return <p id={id} className="mt-1 text-sm font-semibold text-red-800">{message}</p>;
}

export function ProgressBar({ label, percent }) {
  const value = Math.round(Math.min(100, Math.max(0, percent)));
  return (
    <div>
      <div className="flex justify-between text-sm font-semibold text-slate-800">
        <span>{label}</span>
        <span className="tabular-nums">{value}%</span>
      </div>
      <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={value}
        className="mt-1 h-3 overflow-hidden rounded-full bg-slate-200">
        <div className="h-full bg-sky-700 transition-[width]" style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

/**
 * Pick a video and describe what the camera sees, upload it (with progress), then choose
 * "Analyse once" or "Add as live camera". `options` is GET /api/video/options.
 */
export default function VideoUploadForm({ options, onAnalyse, onAddLive, upload = uploadVideo }) {
  const [file, setFile] = useState(null);
  const [form, setForm] = useState(emptyVideoForm);
  const [errors, setErrors] = useState({});
  const [progress, setProgress] = useState(null); // upload percent while uploading
  const [video, setVideo] = useState(null); // the backend's info once uploaded
  const [busy, setBusy] = useState(null); // "analyse" | "live" while that request runs
  const [serverError, setServerError] = useState(null);
  const maxMb = options?.max_upload_mb ?? 500;
  const area = areaFrom(form);

  const set = (field) => (event) => {
    const value = event.target.type === "checkbox" ? event.target.checked : event.target.value;
    setForm((previous) => ({ ...previous, [field]: value }));
  };

  function chooseFile(event) {
    const chosen = event.target.files?.[0] ?? null;
    setFile(chosen);
    setVideo(null);
    setServerError(null);
    setErrors((previous) => ({ ...previous, file: chosen ? fileError(chosen, maxMb) : null }));
    if (chosen && !form.name.trim()) setForm((previous) => ({ ...previous, name: nameFromFile(chosen.name) }));
  }

  function validate(checkFile) {
    const found = { ...zoneErrors(form), ...(checkFile ? { file: fileError(file, maxMb) } : {}) };
    Object.keys(found).forEach((key) => found[key] == null && delete found[key]);
    setErrors(found);
    return Object.keys(found).length === 0;
  }

  async function submitUpload(event) {
    event.preventDefault();
    setServerError(null);
    if (!validate(true)) return;
    setProgress(0);
    try {
      setVideo(await upload(file, setProgress));
    } catch (error) {
      setServerError(error.message);
    } finally {
      setProgress(null);
    }
  }

  async function choose(mode) {
    setServerError(null);
    if (!validate(false)) return;
    setBusy(mode);
    try {
      await (mode === "analyse" ? onAnalyse : onAddLive)(video, zoneRequest(form));
    } catch (error) {
      setServerError(error.message);
    } finally {
      setBusy(null);
    }
  }

  const uploading = progress !== null;
  const disabled = uploading || busy !== null;
  const noModel = options ? !options.model_loaded : false; // never offer counts without trained weights
  return (
    <form onSubmit={submitUpload} noValidate className="space-y-5 rounded-xl bg-white p-4 shadow-sm sm:p-6">
      <div>
        <label htmlFor="video-file" className="font-bold">Video file</label>
        <input id="video-file" type="file" accept={VIDEO_TYPES.join(",")} onChange={chooseFile} disabled={disabled}
          aria-describedby="video-file-help video-file-error" aria-invalid={Boolean(errors.file)}
          className="mt-1 block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-slate-900 file:px-4 file:py-2 file:font-bold file:text-white" />
        <p id="video-file-help" className="mt-1 text-sm text-slate-700">MP4, AVI, MOV or MKV, up to {maxMb} MB.</p>
        <FieldError id="video-file-error" message={errors.file} />
      </div>

      <div>
        <label htmlFor="zone-name" className="font-bold">Zone name</label>
        <input id="zone-name" type="text" value={form.name} onChange={set("name")} maxLength={80}
          aria-invalid={Boolean(errors.name)} aria-describedby="zone-name-error" className={inputClass} />
        <FieldError id="zone-name-error" message={errors.name} />
      </div>

      <fieldset>
        <legend className="font-bold">Visible area (required)</legend>
        <p id="area-help" className="text-sm text-slate-700">
          Approximate ground area the camera sees, e.g. 20 m x 15 m = 300 m². Density and risk depend on this.
        </p>
        <div className="mt-2 flex flex-wrap gap-4 text-sm font-semibold">
          <label className="inline-flex items-center gap-2">
            <input type="radio" name="area-mode" value="area" checked={form.areaMode === "area"} onChange={set("areaMode")} />
            Area in m²
          </label>
          <label className="inline-flex items-center gap-2">
            <input type="radio" name="area-mode" value="size" checked={form.areaMode === "size"} onChange={set("areaMode")} />
            Width × length in metres
          </label>
        </div>
        {form.areaMode === "area" ? (
          <div className="mt-2 max-w-xs">
            <label htmlFor="area" className="text-sm font-semibold">Area (m²)</label>
            <input id="area" type="number" inputMode="decimal" min="0" step="any" value={form.area} onChange={set("area")}
              aria-invalid={Boolean(errors.area)} aria-describedby="area-help area-error" className={inputClass} />
          </div>
        ) : (
          <div className="mt-2 flex max-w-md flex-wrap items-end gap-3">
            <div className="w-32">
              <label htmlFor="area-width" className="text-sm font-semibold">Width (m)</label>
              <input id="area-width" type="number" inputMode="decimal" min="0" step="any" value={form.width}
                onChange={set("width")} aria-invalid={Boolean(errors.area)} aria-describedby="area-help area-error" className={inputClass} />
            </div>
            <div className="w-32">
              <label htmlFor="area-length" className="text-sm font-semibold">Length (m)</label>
              <input id="area-length" type="number" inputMode="decimal" min="0" step="any" value={form.length}
                onChange={set("length")} aria-invalid={Boolean(errors.area)} aria-describedby="area-help area-error" className={inputClass} />
            </div>
            <p className="pb-2 text-sm font-bold tabular-nums" aria-live="polite">
              = {area ? `${area.toLocaleString("en-IN")} m²` : "— m²"}
            </p>
          </div>
        )}
        <FieldError id="area-error" message={errors.area} />
      </fieldset>

      <div className="max-w-md">
        <label htmlFor="location" className="font-bold">Location on the exit map (optional)</label>
        <select id="location" value={form.location} onChange={set("location")} className={inputClass}
          aria-describedby="location-help">
          <option value="">None: no exit advice</option>
          {(options?.locations ?? []).map((node) => <option key={node} value={node}>{prettyNode(node)}</option>)}
        </select>
        <p id="location-help" className="mt-1 text-sm text-slate-700">
          Used only to suggest exits for this video. It never changes the exit advice for the live zones.
        </p>
      </div>

      <div>
        <label className="inline-flex items-start gap-2 font-semibold">
          <input type="checkbox" checked={form.telegram} onChange={set("telegram")} className="mt-1 h-4 w-4"
            disabled={options && !options.telegram_available} aria-describedby="telegram-help" />
          Send Telegram alerts for this video
        </label>
        <p id="telegram-help" className="ml-6 text-sm text-slate-700">
          {options && !options.telegram_available
            ? "Telegram is not set up on the backend (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env)."
            : "Off by default. Messages start with [VIDEO TEST] so nobody mistakes them for a live incident."}
        </p>
      </div>

      {uploading && <ProgressBar label="Uploading" percent={progress} />}
      {serverError && (
        <p role="alert" className="rounded-md border-l-8 border-red-700 bg-red-50 p-3 font-semibold text-red-950">{serverError}</p>
      )}

      {video ? (
        <div className="rounded-lg border-2 border-green-700 bg-green-50 p-4">
          <p className="font-bold text-green-950">
            Uploaded: {video.filename} · {formatDuration(video.duration_s)} · {video.width}×{video.height} ·{" "}
            {video.fps} fps
          </p>
          <p className="mt-1 text-sm text-slate-800">What should happen with it?</p>
          <div className="mt-3 flex flex-wrap gap-3">
            <button type="button" onClick={() => choose("analyse")} disabled={disabled || noModel}
              className="rounded-md bg-slate-900 px-4 py-2 font-bold text-white disabled:opacity-60 focus:outline-none focus-visible:ring-4 focus-visible:ring-sky-500">
              {busy === "analyse" ? "Starting…" : "Analyse once"}
            </button>
            <button type="button" onClick={() => choose("live")} disabled={disabled || noModel}
              className="rounded-md bg-violet-800 px-4 py-2 font-bold text-white disabled:opacity-60 focus:outline-none focus-visible:ring-4 focus-visible:ring-sky-500">
              {busy === "live" ? "Adding…" : "Add as live camera"}
            </button>
          </div>
          <p className="mt-2 text-sm text-slate-700">
            Analyse once: process the whole video and show a report. Add as live camera: loop it as a new zone on
            the live dashboard (remove it below when done).
          </p>
        </div>
      ) : (
        <button type="submit" disabled={disabled}
          className="rounded-md bg-sky-800 px-5 py-2.5 font-bold text-white disabled:opacity-60 focus:outline-none focus-visible:ring-4 focus-visible:ring-sky-400">
          {uploading ? "Uploading…" : "Upload video"}
        </button>
      )}
    </form>
  );
}
