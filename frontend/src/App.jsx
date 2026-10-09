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
import { RedAlertBanner, activeRedAlerts } from "./cc/AlertFeed.jsx";
import CommandCentre from "./cc/CommandCentre.jsx";
import TopBar from "./cc/TopBar.jsx";

// The detail view and the video report carry the chart library; load them only when opened,
// which keeps the first page load light on a guard's phone.
const ZoneDetail = lazy(() => import("./components/ZoneDetail.jsx"));
const VideoReport = lazy(() => import("./components/VideoReport.jsx"));
import { activeBanners } from "./liveState.js";
import { displayLevel, sortZones } from "./risk.js";
import { useAlertSound, useRepeatingSiren } from "./sound.js";
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

/** Everything after login. Live monitoring stays connected on every page, so alerts and banners
 * show up during a video analysis too. #/ is the Command Centre; #/classic the original dashboard. */
function Workspace({ user, onLogout }) {
  const live = useLiveZones();
  const route = useHashRoute();
  const videoJobs = useVideoJobs(user.role === "admin");
  const sound = useAlertSound(live.state.alerts); // siren for RED, tone for ORANGE, each alert once
  const [ccSelected, setCcSelected] = useState(null);
  const allowed = user.role === "admin" || !ADMIN_PAGES.has(route.page);
  const zones = sortZones(Object.values(live.state.zones), live.now, live.connection === "live");
  const zoneNames = Object.fromEntries(zones.map((z) => [z.id, z.name]));
  // An unacknowledged RED alert repeats the siren every 30 s (the classic view keeps its old behaviour).
  useRepeatingSiren(route.page !== "classic" && activeRedAlerts(live.state.alerts, live.now).length > 0, sound);

  if (route.page === "classic") {
    return <ClassicView live={live} user={user} onLogout={onLogout} sound={sound} />;
  }
  return (
    <div className="min-h-screen bg-navy-950 font-body text-slate-100">
      <TopBar connection={live.connection} now={live.now} user={user} onLogout={onLogout} sound={sound} page={route.page} />
      <RedAlertBanner alerts={live.state.alerts} zoneNames={zoneNames} onAcknowledge={live.acknowledge} now={live.now} sound={sound} />
      {route.page === "live" && <CommandCentre live={live} selectedId={ccSelected} onSelect={setCcSelected} />}
      {route.page !== "live" && (
        <div className="min-h-[calc(100vh-4rem)] bg-slate-100 font-sans text-slate-900">
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
        </div>
      )}
    </div>
  );
}

/** The original dashboard, unchanged, as a fallback (#/classic). */
function ClassicView({ live, user, onLogout, sound }) {
  const { state, now, connection, backend, dismissBanner, loadHistory } = live;
  const route = useHashRoute();
  const [selectedId, setSelectedId] = useState(null);
  const [overlay, setOverlay] = useState("none");
  const feed = useZoneFeed(selectedId, overlay);

  const connected = connection === "live";
  const zones = sortZones(Object.values(state.zones), now, connected);
  const zoneNames = Object.fromEntries(zones.map((z) => [z.id, z.name]));
  const selected = selectedId ? state.zones[selectedId] : null;
  const showZones = backend.status === "ok" || backend.health;

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

      <main className="mx-auto grid max-w-7xl gap-6 px-3 py-4 sm:px-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
          <section aria-labelledby="zones-heading">
            <h2 id="zones-heading" className="sr-only">Zones, highest risk first</h2>
            <StatusNotice backend={backend} connection={connection} apiUrl={API_URL} zoneCount={zones.length} />
            {showZones && <ZoneGrid zones={zones} now={now} connected={connected} onSelect={setSelectedId}
              countsAvailable={backend.health?.model_loaded !== false} />}
          </section>
          <AlertPanel alerts={state.alerts} zoneNames={zoneNames} />
      </main>

      {selected && (
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
