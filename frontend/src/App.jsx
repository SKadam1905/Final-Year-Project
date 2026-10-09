import { Suspense, lazy, useCallback, useEffect, useState } from "react";
import { API_URL, authApi, onUnauthorized, session } from "./api.js";
import AlertPanel from "./components/AlertPanel.jsx";
import Header from "./components/Header.jsx";
import LoginPage, { PROJECT_TITLE } from "./components/LoginPage.jsx";
import RedAlertBanners from "./components/RedAlertBanners.jsx";
import ReportsPage from "./components/ReportsPage.jsx";
import StatusNotice from "./components/StatusNotice.jsx";
import VideoPage from "./components/VideoPage.jsx";
import ZoneGrid from "./components/ZoneGrid.jsx";

// The detail view and the video report carry the chart library; load them only when opened,
// which keeps the first page load light on a guard's phone.
const ZoneDetail = lazy(() => import("./components/ZoneDetail.jsx"));
const VideoReport = lazy(() => import("./components/VideoReport.jsx"));
import { activeBanners } from "./liveState.js";
import { displayLevel, sortZones } from "./risk.js";
import { useAlertSound } from "./sound.js";
import { useHashRoute } from "./useHashRoute.js";
import { useLiveZones, useZoneFeed } from "./useLiveZones.js";
import { useVideoJobs } from "./useVideoJobs.js";

const loading = <div className="p-6 font-semibold" aria-busy="true">Loading…</div>;
const ADMIN_PAGES = new Set(["video", "report", "reports"]);

function AdminOnly() {
  return (
    <main className="mx-auto max-w-3xl px-3 py-10 sm:px-6">
      <p role="alert" className="rounded-md border-l-8 border-amber-600 bg-amber-50 p-4 font-semibold text-amber-950">
        This page is for the Control Room Admin. Security Officers see the live dashboard and acknowledge alerts.
      </p>
    </main>
  );
}

/** Everything after login. Live monitoring stays connected on every page, so RED banners show up
 * during a video analysis too. */
function Workspace({ user, onLogout }) {
  const { state, now, connection, backend, dismissBanner, loadHistory } = useLiveZones();
  const route = useHashRoute();
  const videoJobs = useVideoJobs(user.role === "admin");
  const sound = useAlertSound(state.alerts); // siren for RED, tone for ORANGE, each alert once
  const [selectedId, setSelectedId] = useState(null);
  const [overlay, setOverlay] = useState("none");
  const live = route.page === "live" || route.page === "classic";
  const feed = useZoneFeed(live ? selectedId : null, overlay);

  const connected = connection === "live";
  const zones = sortZones(Object.values(state.zones), now, connected);
  const zoneNames = Object.fromEntries(zones.map((z) => [z.id, z.name]));
  const selected = selectedId ? state.zones[selectedId] : null;
  const showZones = backend.status === "ok" || backend.health;
  const allowed = user.role === "admin" || !ADMIN_PAGES.has(route.page);

  useEffect(() => {
    if (selectedId) loadHistory(selectedId);
  }, [selectedId, loadHistory]);

  const closeDetail = useCallback(() => {
    setSelectedId(null);
    setOverlay("none");
  }, []);

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <Header connection={connection} lastUpdate={state.lastUpdate} page={route.page} sound={sound} user={user} onLogout={onLogout} />
      <div className="print:hidden">
        <RedAlertBanners alerts={activeBanners(state)} zoneNames={zoneNames} onDismiss={dismissBanner} />
      </div>

      {!allowed && <AdminOnly />}
      {allowed && route.page === "video" && (
        <VideoPage jobs={videoJobs.list} onJob={videoJobs.upsert} videoZones={zones.filter((z) => z.kind === "video")} />
      )}
      {allowed && route.page === "reports" && <ReportsPage zones={zones} />}
      {allowed && route.page === "report" && (
        <Suspense fallback={loading}>
          <VideoReport jobId={route.jobId} job={videoJobs.jobs[route.jobId]} loaded={videoJobs.loaded} onJob={videoJobs.upsert} />
        </Suspense>
      )}
      {live && (
        <main className="mx-auto grid max-w-7xl gap-6 px-3 py-4 sm:px-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
          <section aria-labelledby="zones-heading">
            <h2 id="zones-heading" className="sr-only">Zones, highest risk first</h2>
            <StatusNotice backend={backend} connection={connection} apiUrl={API_URL} zoneCount={zones.length} />
            {showZones && <ZoneGrid zones={zones} now={now} connected={connected} onSelect={setSelectedId}
              countsAvailable={backend.health?.model_loaded !== false} />}
          </section>
          <AlertPanel alerts={state.alerts} zoneNames={zoneNames} />
        </main>
      )}

      {live && selected && (
        <Suspense fallback={<div className="fixed inset-0 z-30 bg-slate-900/70" aria-busy="true" />}>
          <ZoneDetail
            zone={selected}
            level={displayLevel(selected, now, connected)}
            feed={feed}
            overlay={overlay}
            onOverlayChange={setOverlay}
            thresholds={backend.config?.thresholds}
            densityScale={backend.config?.density_scale}
            onClose={closeDetail}
          />
        </Suspense>
      )}

      {backend.health?.model && (
        <footer className="mx-auto max-w-7xl px-3 pb-6 text-xs text-slate-700 sm:px-6">
          Crowd model: {backend.health.model.label}
        </footer>
      )}
    </div>
  );
}

/** Login gate: the login page until there is a valid session; back to it on logout, on any 401
 * (expired or revoked token) and when the 8 h session runs out while the page is open. */
export default function App() {
  const [auth, setAuth] = useState(() => session.get());
  const [notice, setNotice] = useState(null);

  useEffect(() => {
    document.title = PROJECT_TITLE;
    onUnauthorized((message) => {
      setAuth(null);
      setNotice(message || "Please log in again.");
    });
  }, []);

  useEffect(() => {
    if (!auth) return undefined;
    const timer = setTimeout(() => {
      session.clear();
      setAuth(null);
      setNotice("Your session has expired: please log in again.");
    }, Math.max(0, auth.expires_at * 1000 - Date.now()));
    return () => clearTimeout(timer);
  }, [auth]);

  const logout = useCallback(async () => {
    await authApi.logout();
    setAuth(null);
    setNotice("You have logged out.");
  }, []);

  if (!auth) {
    return <LoginPage notice={notice} onLogin={(value) => { setNotice(null); setAuth(value); }} />;
  }
  return <Workspace key={auth.token} user={auth.user} onLogout={logout} />;
}
