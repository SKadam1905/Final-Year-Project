import { formatTime } from "../format.js";
import { routeHref } from "../useHashRoute.js";

const CONNECTION = {
  live: { text: "Live", icon: "●", style: "bg-green-700 text-white" },
  connecting: { text: "Connecting…", icon: "…", style: "bg-slate-200 text-slate-900" },
  disconnected: { text: "Disconnected", icon: "✕", style: "bg-red-700 text-white" },
};

const PAGES = [
  { page: "command", href: routeHref.live, label: "Command Centre" },
  { page: "classic", href: routeHref.classic, label: "Live dashboard" },
  { page: "video", href: routeHref.video, label: "Video analysis", admin: true },
  { page: "reports", href: routeHref.reports, label: "Reports", admin: true },
];

const BUTTON = "rounded-md px-3 py-1.5 text-sm font-bold focus:outline-none focus-visible:ring-4 focus-visible:ring-sky-400";

/** "Enable sound" until clicked (browsers need a click before playing audio), then a mute toggle. */
function SoundControl({ sound }) {
  if (!sound?.supported) return null;
  if (!sound.enabled) {
    return (
      <button type="button" onClick={sound.enable} className={`${BUTTON} bg-amber-400 text-black hover:bg-amber-300`}>
        <span aria-hidden="true">🔔 </span>Enable sound
      </button>
    );
  }
  return (
    <button type="button" onClick={sound.toggleMute} aria-pressed={sound.muted}
      title={sound.muted ? "Alert sounds are muted" : "A siren plays for RED alerts, a short tone for ORANGE"}
      className={`${BUTTON} ${sound.muted ? "bg-slate-600 text-white hover:bg-slate-500" : "bg-white text-slate-900"}`}>
      <span aria-hidden="true">{sound.muted ? "🔇 " : "🔊 "}</span>
      {sound.muted ? "Sound muted" : "Sound on"}
    </button>
  );
}

export default function Header({ connection, lastUpdate, page = "live", sound = null, user = null, onLogout = null }) {
  const status = CONNECTION[connection];
  const current = page === "report" ? "video" : page === "live" ? "classic" : page;
  const pages = PAGES.filter((item) => !item.admin || !user || user.role === "admin");
  return (
    <header className="sticky top-0 z-20 bg-slate-900 text-white shadow print:hidden">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-3 py-3 sm:px-6">
        <div>
          <h1 className="text-lg font-bold leading-tight sm:text-2xl">Ramkund crowd safety</h1>
          <p className="text-xs text-slate-300 sm:text-sm">Nashik Kumbh Mela 2027 · stampede risk prototype</p>
        </div>
        <nav aria-label="Pages" className="flex gap-2">
          {pages.map((item) => (
            <a key={item.page} href={item.href} aria-current={current === item.page ? "page" : undefined}
              className={`rounded-md px-3 py-1.5 text-sm font-bold focus:outline-none focus-visible:ring-4 focus-visible:ring-sky-400 ${
                current === item.page ? "bg-white text-slate-900" : "bg-slate-700 text-white hover:bg-slate-600"
              }`}>
              {item.label}
            </a>
          ))}
        </nav>
        <SoundControl sound={sound} />
        <div className="flex flex-wrap items-center gap-3" role="status" aria-live="polite">
          <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-bold ${status.style}`}>
            <span aria-hidden="true">{status.icon}</span>
            {status.text}
          </span>
          <span className="text-sm text-slate-200">
            Last update <time className="font-semibold text-white">{formatTime(lastUpdate)}</time>
          </span>
        </div>
        {user && (
          <div className="flex items-center gap-2 text-sm">
            <span className="text-slate-200"><span className="font-semibold text-white">{user.name}</span> · {user.role_label}</span>
            {onLogout && (
              <button type="button" onClick={onLogout} className={`${BUTTON} bg-slate-700 text-white hover:bg-slate-600`}>Log out</button>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
