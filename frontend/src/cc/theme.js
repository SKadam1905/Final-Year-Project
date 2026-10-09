// Command Centre styles for each displayed level (risk.js decides the level). Colour is never the
// only cue: every level also has a word and an icon. Chip text colours keep WCAG AA contrast.

export const LEVEL_UI = {
  RED: { word: "RED", meaning: "Danger", icon: "■", fill: "#EF4444", chip: "bg-risk-red text-white", text: "text-red-300", rank: 5 },
  ORANGE: { word: "ORANGE", meaning: "Warning", icon: "◆", fill: "#FB923C", chip: "bg-risk-orange text-navy-950", text: "text-orange-300", rank: 4 },
  "NO SIGNAL": { word: "NO SIGNAL", meaning: "No recent data", icon: "✕", fill: "#64748B", chip: "bg-slate-600 text-white", text: "text-slate-300", rank: 3 },
  YELLOW: { word: "YELLOW", meaning: "Watch", icon: "▲", fill: "#FACC15", chip: "bg-risk-yellow text-navy-950", text: "text-yellow-200", rank: 2 },
  GREEN: { word: "GREEN", meaning: "Normal", icon: "●", fill: "#22C55E", chip: "bg-risk-green text-navy-950", text: "text-green-300", rank: 1 },
  WAITING: { word: "WAITING", meaning: "No reading yet", icon: "…", fill: "#334155", chip: "bg-navy-600 text-slate-100", text: "text-slate-300", rank: 0 },
};

export const levelUi = (label) => LEVEL_UI[label] ?? LEVEL_UI.WAITING;

/** Pressure (or density) -> the level band it falls in, for walkway colours and chart bands. */
export function bandFor(value, thresholds) {
  if (value == null || !thresholds) return "GREEN";
  if (value >= thresholds.red) return "RED";
  if (value >= thresholds.orange) return "ORANGE";
  if (value >= thresholds.yellow) return "YELLOW";
  return "GREEN";
}

/** A short name for the active crowd model, from /health. */
export function modelName(health) {
  const model = health?.model;
  if (!model) return health?.model_loaded === false ? "No trained model" : "—";
  const base = {
    public_pretrained: `CSRNet · public weights (${model.dataset ?? "ShanghaiTech"})`,
    trained: `CSRNet · trained here (${model.dataset ?? ""})`,
    demo: "Demo model (not real counts)",
    untrained: "Untrained weights",
  }[model.kind] ?? "CSRNet";
  const detector = health?.detector?.model ? ` + ${health.detector.model.replace(/^yolov8/i, "YOLOv8")}` : "";
  return base + detector;
}
