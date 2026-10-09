import { useEffect, useRef, useState } from "react";
import { authApi } from "../api.js";

export const PROJECT_TITLE = "Nashik Kumbh Mela 2027 · Crowd Safety Command Centre";
export const TEAM_CREDIT = [
  "Group 27 · Shubham Kadam · Rohit Kokare · Shraddha Deokar · Mitali Mishra",
  "Guide: Mr. A. S. Dalvi · Department of Information Technology, Sanjivani College of Engineering, Kopargaon · AY 2026–27",
];

/** Decorative only: the river and the ghat steps as line art, behind the sign-in card. */
function RiverArt() {
  return (
    <svg aria-hidden="true" viewBox="0 0 800 600" preserveAspectRatio="xMidYMid slice"
      className="pointer-events-none absolute inset-0 h-full w-full opacity-[0.16]">
      <path d="M-20 420 C 160 340, 300 470, 460 380 S 720 300, 840 360" fill="none" stroke="#F2A54A" strokeWidth="2" />
      <path d="M-20 450 C 160 370, 300 500, 460 410 S 720 330, 840 390" fill="none" stroke="#F2A54A" strokeWidth="1" />
      {Array.from({ length: 7 }, (_, i) => (
        <path key={i} d={`M ${60 + i * 12} ${560 - i * 12} h ${200 - i * 18}`} stroke="#9FB3D9" strokeWidth="1.5" />
      ))}
      <circle cx="620" cy="140" r="60" fill="none" stroke="#9FB3D9" strokeWidth="1" />
      <circle cx="620" cy="140" r="90" fill="none" stroke="#9FB3D9" strokeWidth="0.6" />
    </svg>
  );
}

/** Full-screen sign-in. Accounts come from the control room (backend/tools/create_user.py). */
export default function LoginPage({ notice = null, onLogin, login = authApi.login }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const userField = useRef(null);

  useEffect(() => userField.current?.focus(), []);

  async function submit(event) {
    event.preventDefault();
    if (!username.trim() || !password) {
      setError("Enter your username and password.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const value = await login(username.trim(), password);
      onLogin(value);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  const field = "mt-1 block w-full rounded-lg border-2 border-navy-500 bg-navy-950 px-3 py-2.5 text-base text-white "
    + "placeholder:text-slate-500 focus:border-saffron focus:outline-none focus:ring-2 focus:ring-saffron/40";

  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden bg-navy-950 font-body text-slate-100">
      <RiverArt />
      <div className="relative mx-auto flex w-full max-w-6xl flex-1 flex-col items-center justify-center gap-10 px-4 py-10 lg:flex-row lg:justify-between lg:px-10">
        <header className="max-w-xl text-center lg:text-left">
          <p className="font-display text-sm font-semibold uppercase tracking-[0.25em] text-saffron">Nashik Kumbh Mela 2027</p>
          <h1 className="mt-3 font-display text-3xl font-bold leading-tight text-white sm:text-5xl">
            Crowd Safety<br className="hidden sm:block" /> Command Centre
          </h1>
          <p className="mt-4 text-lg text-slate-300">
            Live crowd density, turbulence and stampede risk for the Ramkund ghats, with the safest exit for every zone.
          </p>
          <p className="mt-2 text-sm text-slate-400">For authorised control-room and security staff only.</p>
        </header>

        <form onSubmit={submit} noValidate aria-labelledby="signin-heading"
          className="w-full max-w-sm rounded-2xl border border-navy-600 bg-navy-800/95 p-6 shadow-2xl shadow-black/40 backdrop-blur sm:p-8">
          <h2 id="signin-heading" className="font-display text-2xl font-semibold text-white">Sign in</h2>
          {notice && (
            <p role="status" className="mt-3 rounded-lg border-l-4 border-saffron bg-saffron/10 px-3 py-2 text-sm text-saffron-300">{notice}</p>
          )}
          <label className="mt-5 block text-sm font-medium text-slate-200">
            Username
            <input ref={userField} value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username"
              autoCapitalize="none" spellCheck={false} className={field} />
          </label>
          <label htmlFor="password" className="mt-4 block text-sm font-medium text-slate-200">Password</label>
          <div className="relative">
            <input id="password" type={showPassword ? "text" : "password"} value={password}
              onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" className={`${field} pr-20`} />
            <button type="button" onClick={() => setShowPassword((v) => !v)} aria-pressed={showPassword}
              aria-label={showPassword ? "Hide password" : "Show password"}
              className="absolute inset-y-0 right-1 my-1 rounded-md px-3 text-sm font-semibold text-saffron hover:bg-navy-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-saffron">
              {showPassword ? "Hide" : "Show"}
            </button>
          </div>
          {error && (
            <p role="alert" className="mt-4 rounded-lg border-l-4 border-risk-red bg-risk-red/10 px-3 py-2 text-sm font-medium text-red-200">{error}</p>
          )}
          <button type="submit" disabled={busy}
            className="mt-6 flex w-full items-center justify-center gap-2 rounded-lg bg-saffron px-4 py-3 font-display text-lg font-semibold text-navy-950 transition hover:bg-saffron-300 focus:outline-none focus-visible:ring-4 focus-visible:ring-saffron/50 disabled:cursor-wait disabled:opacity-70">
            {busy && <span aria-hidden="true" className="h-4 w-4 animate-spin rounded-full border-2 border-navy-950 border-t-transparent" />}
            {busy ? "Signing in…" : "Sign in"}
          </button>
          <p className="mt-4 text-xs text-slate-400">Accounts are issued by the control room administrator.</p>
        </form>
      </div>
      <footer className="relative border-t border-navy-700 px-4 py-4 text-center text-xs leading-relaxed text-slate-400">
        {TEAM_CREDIT.map((line) => <p key={line}>{line}</p>)}
      </footer>
    </div>
  );
}
