const KINDS = {
  trained: { title: "Model: CSRNet trained on this project", tone: "border-slate-500 bg-white text-slate-900" },
  public_pretrained: { title: "Model: public pretrained CSRNet", tone: "border-sky-700 bg-sky-50 text-sky-950" },
  demo: { title: "Demo model: counts are not real estimates", tone: "border-orange-500 bg-orange-50 text-orange-950" },
  untrained: { title: "Test model: counts are not meaningful", tone: "border-orange-500 bg-orange-50 text-orange-950" },
};

/** Which weights produced the counts: a neutral note for real weights, a warning for demo or test weights. */
export default function ModelNotice({ model }) {
  if (!model) return null;
  const kind = KINDS[model.kind] ?? KINDS.untrained;
  return (
    <div role="note" className={`rounded-xl border-l-8 p-4 ${kind.tone}`}>
      <p className="font-bold">{kind.title}</p>
      <p className="mt-1 text-sm">{model.label}{model.weights ? ` (weights: ${model.weights})` : ""}</p>
    </div>
  );
}
