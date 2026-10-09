// REST client for backend/app/main.py. The contract below was read from that code:
//
//   Every /api/ call needs the session token from POST /api/auth/login (backend/app/auth.py):
//   "Authorization: Bearer <token>", or ?token=<token> on <img> / download links. A 401 means
//   not logged in or expired: the session is cleared and the app returns to the login page.
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

// The session lives in sessionStorage: it survives a page reload but not closing the tab.
const SESSION_KEY = "kumbh.session";

export const session = {
  /** { token, expires_at, user: { username, name, role, role_label } } or null (also once expired). */
  get() {
    try {
      const value = JSON.parse(window.sessionStorage.getItem(SESSION_KEY));
      return value?.token && value.expires_at * 1000 > Date.now() ? value : null;
    } catch {
      return null;
    }
  },
  set(value) {
    try {
      window.sessionStorage.setItem(SESSION_KEY, JSON.stringify(value));
    } catch {
      // storage blocked: the session lasts until the page is reloaded
    }
    memorySession = value;
  },
  clear() {
    try {
      window.sessionStorage.removeItem(SESSION_KEY);
    } catch {
      // nothing stored
    }
    memorySession = null;
  },
  token() {
    return (this.get() ?? memorySession)?.token ?? null;
  },
};
let memorySession = null;

let unauthorized = () => {};
/** Called with the backend's reason whenever a request comes back 401 (logged out or expired). */
export function onUnauthorized(handler) {
  unauthorized = handler;
}

function authHeaders(extra = {}) {
  const token = session.token();
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

/** A URL for <img src> or a download link, which cannot send the Authorization header. */
export function withToken(url) {
  const token = session.token();
  return token ? `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}` : url;
}

async function failed(response) {
  const detail = await errorDetail(response);
  if (response.status === 401) {
    session.clear();
    unauthorized(detail);
  }
  return new Error(detail);
}

async function errorDetail(response) {
  try {
    const body = await response.json();
    return body.detail ?? `HTTP ${response.status}`;
  } catch {
    return `HTTP ${response.status}`;
  }
}

async function getJson(path) {
  const response = await fetch(`${API_URL}${path}`, { headers: authHeaders() });
  if (!response.ok) throw await failed(response);
  return response.json();
}

async function sendJson(method, path, body) {
  const response = await fetch(`${API_URL}${path}`, {
    method,
    headers: authHeaders(body === undefined ? {} : { "Content-Type": "application/json" }),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw await failed(response);
  return response.status === 204 ? null : response.json();
}

// Login and logout (backend/app/auth.py). login() resolves to the stored session.
export const authApi = {
  async login(username, password) {
    let response;
    try {
      response = await fetch(`${API_URL}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
    } catch {
      throw new Error(`Cannot reach the backend at ${API_URL}. Is it running?`);
    }
    if (!response.ok) throw new Error(await errorDetail(response));
    const value = await response.json();
    session.set(value);
    return value;
  },
  async logout() {
    try {
      await fetch(`${API_URL}/api/auth/logout`, { method: "POST", headers: authHeaders() });
    } catch {
      // backend down: the token simply expires
    }
    session.clear();
  },
};

const zonePath = (zoneId) => `/api/zones/${encodeURIComponent(zoneId)}`;

export const api = {
  health: () => getJson("/health"),
  config: () => getJson("/api/config"),
  zones: () => getJson("/api/zones"),
  recentAlerts: (limit = 50) => getJson(`/api/alerts/recent?limit=${limit}`),
  history: (zoneId, limit = 150) => getJson(`${zonePath(zoneId)}/history?limit=${limit}`),
  siteMap: () => getJson("/api/site-map"),
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

export const jobUrl = (jobId, file) => withToken(`${API_URL}${jobPath(jobId)}/${file}`);
export const sampleUrl = (jobId, index, file) => withToken(`${API_URL}${jobPath(jobId)}/samples/${index}/${file}`);

/**
 * Upload a video as the raw request body, reporting progress (0-100) as it goes. fetch() cannot
 * report upload progress, so this uses XMLHttpRequest. Resolves to the backend's video info.
 */
export function uploadVideo(file, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/api/videos?filename=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    const token = session.token();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.upload.onprogress = (event) => event.lengthComputable && onProgress((100 * event.loaded) / event.total);
    xhr.onload = () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        // not JSON
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else {
        if (xhr.status === 401) {
          session.clear();
          unauthorized(body?.detail);
        }
        reject(new Error(body?.detail ?? `Upload failed (HTTP ${xhr.status})`));
      }
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
  const response = await fetch(url, { cache: "no-store", headers: authHeaders() });
  if (!response.ok) throw await failed(response);
  const timestamp = Number(response.headers.get("X-Frame-Timestamp"));
  return { url: URL.createObjectURL(await response.blob()), timestamp: timestamp > 0 ? timestamp : null };
}

/** Socket.IO options: the token is read again on every (re)connect. */
export const socketOptions = (extra = {}) => ({
  transports: ["websocket", "polling"],
  autoConnect: false,
  auth: (cb) => cb({ token: session.token() }),
  ...extra,
});

/** A refused socket (no or expired token) logs out like a 401; other errors are just "disconnected". */
export function socketRefused(error) {
  if (/log in|logged out/i.test(error?.message ?? "")) {
    session.clear();
    unauthorized(error.message);
    return true;
  }
  return false;
}
