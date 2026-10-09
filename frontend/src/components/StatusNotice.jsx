const TONES = {
  error: "border-red-700 bg-red-50 text-red-950",
  warning: "border-orange-500 bg-orange-50 text-orange-950",
  info: "border-slate-400 bg-white text-slate-900",
};

function Notice({ tone, title, children }) {
  return (
    <div role={tone === "error" ? "alert" : "status"} className={`mb-4 rounded-xl border-l-8 p-4 shadow-sm ${TONES[tone]}`}>
      <p className="font-bold">{title}</p>
      {children && <div className="mt-1 text-sm">{children}</div>}
    </div>
  );
}

/** Empty and error states: backend unreachable or lost, model not loaded, no zones configured. */
export default function StatusNotice({ backend, connection, apiUrl, zoneCount }) {
  if (backend.status === "loading") {
    return <Notice tone="info" title="Connecting to the backend…" />;
  }
  if (backend.status === "unreachable" && !backend.health) {
    return (
      <Notice tone="error" title={`Cannot reach the backend at ${apiUrl}`}>
        <p>
          Retrying every 5 seconds. Is it running? From the backend folder:{" "}
          <code>uvicorn app.main:app --port 8000</code>
        </p>
        {backend.error && <p className="mt-1">Last error: {backend.error}</p>}
      </Notice>
    );
  }
  return (
    <>
      {connection === "disconnected" && (
        <Notice tone="error" title="Connection to the backend lost">
          <p>Reconnecting automatically. Risk levels are hidden until live data returns.</p>
        </Notice>
      )}
      {backend.health && !backend.health.model_loaded && (
        <Notice tone="warning" title="No trained model: counts unavailable">
          <p>{backend.health.model_error}</p>
          <p className="mt-1">Camera images still work. Set CSRNET_WEIGHTS in .env and restart the backend.</p>
        </Notice>
      )}
      {zoneCount === 0 && (
        <Notice tone="info" title="No zones configured">
          <p>Add cameras to <code>config/zones.json</code> and restart the backend.</p>
        </Notice>
      )}
    </>
  );
}
