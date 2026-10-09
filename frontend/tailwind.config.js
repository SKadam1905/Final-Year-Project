/** Tailwind CSS v3 (pinned: v4 dropped `npx tailwindcss init` and this config format). */
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      // Command Centre (dark control-room theme); the classic view keeps Tailwind's slate palette.
      colors: {
        navy: { 950: "#0B1220", 900: "#0F192D", 850: "#121D35", 800: "#14213D", 700: "#1D2D50", 600: "#2A3D66", 500: "#3B5182" },
        saffron: { 300: "#F8C98F", DEFAULT: "#F2A54A", 600: "#D98A2B" },
        risk: { green: "#22C55E", yellow: "#FACC15", orange: "#FB923C", red: "#EF4444", none: "#64748B" },
      },
      fontFamily: {
        display: ['"Rubik"', "system-ui", "sans-serif"],
        body: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
      },
      keyframes: {
        "risk-pulse": { "0%": { transform: "scale(1)", opacity: "0.75" }, "100%": { transform: "scale(2.2)", opacity: "0" } },
        "route-dash": { to: { strokeDashoffset: "-24" } },
        "fade-in": { from: { opacity: "0", transform: "translateY(4px)" }, to: { opacity: "1", transform: "none" } },
      },
      animation: {
        "risk-pulse": "risk-pulse 1.6s ease-out infinite",
        "risk-pulse-fast": "risk-pulse 0.9s ease-out infinite",
        "route-dash": "route-dash 0.8s linear infinite",
        "fade-in": "fade-in 0.25s ease-out",
      },
    },
  },
  plugins: [],
};
