// REST client for backend/app/main.py. The contract below was read from that code:
//
//   GET /health                 { model_loaded, model_error, zones: [id], storage, frame_source,
//                                 cameras: { id: {...} | null }, server_time }
//   GET /api/config             { thresholds: { yellow, orange, red }, burst_interval_s, ... }
//   GET /api/zones              [{ id, name, graph_node, area_m2, calibrated, source, roi }]
//   GET /api/zones/:id/history  { zone_id, server_time, readings: [{ timestamp, pressure, ... }] }
//   GET /api/zones/:id/snapshot.jpg | density.png | detections.png | gradcam.png   (503 + { detail } when unavailable)
//   GET /api/alerts/recent      [{ id, zone_id, risk_level, message, timestamp, delivered, deliveries }]
//
// Socket.IO: "density_update" { zone_id, name, timestamp, server_time, density, turbulence,
// pressure (30 s mean), raw_pressure, risk_level, count, samples, exits: [{ exit, cost, path }] }
// and "alert" (same shape as /api/alerts/recent). Times are Unix epoch seconds; zones are
// identified by `id` (= zone_id, e.g. "RAMKUND_NORTH").

export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

async function errorDetail(response) {
  try {
    const body = await response.json();
    return body.detail ?? `HTTP ${response.status}`;
  } catch {
    return `HTTP ${response.status}`;
  }
}

async function getJson(path) {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) throw new Error(await errorDetail(response));
  return response.json();
}

async function sendJson(method, path, body) {
  const response = await fetch(`${API_URL}${path}`, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  return response.status === 204 ? null : response.json();
}

const zonePath = (zoneId) => `/api/zones/${encodeURIComponent(zoneId)}`;

export const api = {
  health: () => getJson("/health"),
  config: () => getJson("/api/config"),
  zones: () => getJson("/api/zones"),
  recentAlerts: (limit = 50) => getJson(`/api/alerts/recent?limit=${limit}`),
  history: (zoneId, limit = 150) => getJson(`${zonePath(zoneId)}/history?limit=${limit}`),
};

// Video analysis (backend/app/video_api.py). A job is
//   { id, video_id, video_name, name, area_m2, graph_node, telegram_alerts, created_at,
//     state: "queued" | "running" | "done" | "failed", progress (0-100), error,
//     annotated: { state: "none" | "queued" | "running" | "done" | "failed", progress, error } }
// and arrives over Socket.IO as "video_job" while it changes.
const jobPath = (jobId) => `/api/jobs/${encodeURIComponent(jobId)}`;

export const videoApi = {
  options: () => getJson("/api/video/options"),
  analyse: (videoId, body) => sendJson("POST", `/api/videos/${encodeURIComponent(videoId)}/analyse`, body),
  addLive: (videoId, body) => sendJson("POST", `/api/videos/${encodeURIComponent(videoId)}/live`, body),
  removeZone: (zoneId) => sendJson("DELETE", zonePath(zoneId)),
  jobs: () => getJson("/api/jobs"),
  job: (jobId) => getJson(jobPath(jobId)),
  results: (jobId) => getJson(`${jobPath(jobId)}/results`),
  deleteJob: (jobId) => sendJson("DELETE", jobPath(jobId)),
  requestAnnotated: (jobId) => sendJson("POST", `${jobPath(jobId)}/annotated`),
};

// Reports (backend/app/reports.py): filtered history and per-zone summaries; `query` from reports.js.
export const reportsApi = {
  alerts: (query) => getJson(`/api/reports/alerts${query}`),
  incidents: (query) => getJson(`/api/reports/incidents${query}`),
  summary: (query) => getJson(`/api/reports/summary${query}`),
};

export const jobUrl = (jobId, file) => `${API_URL}${jobPath(jobId)}/${file}`;
export const sampleUrl = (jobId, index, file) => `${API_URL}${jobPath(jobId)}/samples/${index}/${file}`;

/**
 * Upload a video as the raw request body, reporting progress (0-100) as it goes. fetch() cannot
 * report upload progress, so this uses XMLHttpRequest. Resolves to the backend's video info.
 */
export function uploadVideo(file, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/api/videos?filename=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (event) => event.lengthComputable && onProgress((100 * event.loaded) / event.total);
    xhr.onload = () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        // not JSON
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body?.detail ?? `Upload failed (HTTP ${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Upload failed: the backend could not be reached."));
    xhr.send(file);
  });
}

const IMAGE_FILES = { snapshot: "snapshot.jpg", density: "density.png", detections: "detections.png", gradcam: "gradcam.png" };

export function imageUrl(zoneId, kind) {
  return `${API_URL}${zonePath(zoneId)}/${IMAGE_FILES[kind]}`;
}

/** Fetch an image as an object URL, with the frame's capture time (epoch seconds). */
export async function fetchImage(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(await errorDetail(response));
  const timestamp = Number(response.headers.get("X-Frame-Timestamp"));
  return { url: URL.createObjectURL(await response.blob()), timestamp: timestamp > 0 ? timestamp : null };
}
