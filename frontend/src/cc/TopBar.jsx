import { routeHref } from "../useHashRoute.js";

const CONNECTION = {
  live: { text: "LIVE", dot: "bg-risk-green", style: "border-risk-green/60 text-green-200" },
  connecting: { text: "CONNECTING", dot: "bg-slate-300", style: "border-slate-500 text-slate-200" },
  disconnected: { text: "DISCONNECTED", dot: "bg-risk-red", style: "border-risk-red bg-risk-red/15 text-red-200" },
};

export const NAV = [
  { page: "live", href: routeHref.live, label: "Command Centre" },
  { page: "video", href: routeHref.video, label: "Video analysis", admin: true },
  { page: "reports", href: routeHref.reports, label: "Reports", admin: true },
  { page: "classic", href: routeHref.classic, label: "Classic view" },
];

const clock = new Intl.DateTimeFormat("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
const day = new Intl.DateTimeFormat("en-IN", { weekday: "short", day: "numeric", month: "short" });
const BUTTON = "rounded-md px-3 py-1.5 text-sm font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-saffron";

function SoundButton({ sound }) {
  if (!sound?.supported) return null;
  if (!sound.enabled) {
    return (
      <button type="button" onClick={sound.enable} className={`${BUTTON} bg-saffron text-navy-950 hover:bg-saffron-300`}>
        <span aria-hidden="true">🔔 </span>Enable sound
      </button>
    );
  }
  return (
    <button type="button" onClick={sound.toggleMute} aria-pressed={sound.muted}
      title={sound.muted ? "Alert sounds are muted" : "Siren for RED (repeats until acknowledged), tone for ORANGE"}
      className={`${BUTTON} border ${sound.muted ? "border-slate-500 text-slate-300" : "border-saffron/60 text-saffron"}`}>
      <span aria-hidden="true">{sound.muted ? "🔇 " : "🔊 "}</span>{sound.muted ? "Sound muted" : "Sound on"}
    </button>
  );
}

/** Command Centre top bar: title, connection, clock, navigation, sound, user and logout. */
export default function TopBar({ connection, now, user, onLogout, sound, page }) {
  const status = CONNECTION[connection] ?? CONNECTION.connecting;
  const current = page === "report" ? "video" : page;
  return (
    <header className="sticky top-0 z-30 border-b border-navy-700 bg-navy-950/95 font-body text-slate-100 backdrop-blur print:hidden">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2 sm:px-4">
        <div className="flex min-w-0 items-center gap-2.5">
          <span aria-hidden="true" className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-saffron font-display text-lg font-bold text-navy-950">K</span>
          <div className="min-w-0">
            <p className="font-display text-[11px] font-semibold uppercase tracking-[0.2em] text-saffron">Nashik Kumbh Mela 2027</p>
            <h1 className="truncate font-display text-base font-semibold leading-tight text-white xl:text-lg">Crowd Safety Command Centre</h1>
          </div>
        </div>
        <nav aria-label="Views" className="order-last flex w-full flex-wrap gap-1 lg:order-none lg:w-auto">
          {NAV.filter((item) => !item.admin || user.role === "admin").map((item) => (
            <a key={item.page} href={item.href} aria-current={current === item.page ? "page" : undefined}
              className={`${BUTTON} ${current === item.page ? "bg-navy-700 text-white shadow-inner ring-1 ring-saffron/50" : "text-slate-300 hover:bg-navy-800 hover:text-white"}`}>
              {item.label}
            </a>
          ))}
        </nav>
        <div className="ml-auto flex flex-wrap items-center gap-2 sm:gap-3">
          <span role="status" aria-live="polite"
            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 font-display text-xs font-semibold tracking-wider ${status.style}`}>
            <span aria-hidden="true" className={`h-2.5 w-2.5 rounded-full ${status.dot} ${connection === "live" ? "animate-pulse" : ""}`} />
            {status.text}
          </span>
          <time className="text-right font-display leading-tight tabular-nums" dateTime={new Date(now).toISOString()}>
            <span className="block text-lg font-semibold text-white">{clock.format(now)}</span>
            <span className="block text-[11px] text-slate-400">{day.format(now)} · IST</span>
          </time>
          <SoundButton sound={sound} />
          <div className="hidden text-right leading-tight sm:block" title={`${user.name} (${user.username}), ${user.role_label}`}>
            <span className="block text-sm font-semibold text-white">{user.name === user.role_label ? user.username : user.name}</span>
            <span className="block text-[11px] text-slate-400">{user.role_label}</span>
          </div>
          <button type="button" onClick={onLogout} className={`${BUTTON} border border-navy-500 text-slate-200 hover:bg-navy-800`}>Log out</button>
        </div>
      </div>
    </header>
  );
}
