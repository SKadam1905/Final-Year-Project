// Upload-form rules for the Video analysis page, kept apart from the component so they are
// easy to test. The backend checks the same things again (and opens the file with OpenCV).

export const VIDEO_TYPES = [".mp4", ".avi", ".mov", ".mkv"];
export const MAX_NAME_LENGTH = 60;
export const MAX_AREA_M2 = 1_000_000;

export const emptyVideoForm = {
  name: "",
  areaMode: "area", // "area" | "size"
  area: "",
  width: "",
  length: "",
  location: "",
  telegram: false,
};

function positive(value) {
  const n = Number(String(value).trim().replace(",", "."));
  return String(value).trim() !== "" && Number.isFinite(n) && n > 0 ? n : null;
}

/** The area in m²: typed directly, or width x length. null when it cannot be worked out. */
export function areaFrom(form) {
  if (form.areaMode === "size") {
    const width = positive(form.width);
    const length = positive(form.length);
    return width && length ? Math.round(width * length * 100) / 100 : null;
  }
  return positive(form.area);
}

export function fileError(file, maxMb) {
  if (!file) return "Choose a video file.";
  const dot = file.name.lastIndexOf(".");
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
  if (!VIDEO_TYPES.includes(ext)) return `Unsupported file type: choose an MP4, AVI, MOV or MKV video.`;
  if (file.size > maxMb * 1024 * 1024) return `The file is ${(file.size / 1024 / 1024).toFixed(0)} MB; the limit is ${maxMb} MB.`;
  if (file.size === 0) return "The file is empty.";
  return null;
}

/** Problems with the zone fields, by field name; empty when the form can be submitted. */
export function zoneErrors(form) {
  const errors = {};
  const name = form.name.trim();
  if (!name) errors.name = "Give the zone a name.";
  else if (name.length > MAX_NAME_LENGTH) errors.name = `Use at most ${MAX_NAME_LENGTH} characters.`;
  const area = areaFrom(form);
  if (area === null) {
    if (form.areaMode === "size") errors.area = "Enter the width and the length in metres (both above 0).";
    else errors.area = "Enter the visible area in square metres (above 0).";
  } else if (area > MAX_AREA_M2) {
    errors.area = `That is more than ${MAX_AREA_M2.toLocaleString("en-IN")} m²; check the numbers.`;
  }
  return errors;
}

/** The body for POST /api/videos/{id}/analyse and /live. */
export function zoneRequest(form) {
  return {
    name: form.name.trim(),
    area_m2: areaFrom(form),
    graph_node: form.location || null,
    telegram_alerts: Boolean(form.telegram),
  };
}

/** "clip_01 final.mp4" -> "clip_01 final": a starting zone name. */
export function nameFromFile(fileName) {
  const dot = fileName.lastIndexOf(".");
  return (dot > 0 ? fileName.slice(0, dot) : fileName).slice(0, MAX_NAME_LENGTH);
}
