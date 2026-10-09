import { useEffect, useState } from "react";

/**
 * Pages by URL hash, so no router library is needed and the browser's back button works:
 *   #/ (or nothing) -> { page: "live" }   #/video -> { page: "video" }   #/video/<jobId> -> { page: "report", jobId }
 *   #/reports -> { page: "reports" }   #/classic -> { page: "classic" } (the original dashboard)
 */
export function parseRoute(hash) {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (parts[0] === "reports") return { page: "reports" };
  if (parts[0] === "classic") return { page: "classic" };
  if (parts[0] !== "video") return { page: "live" };
  return parts[1] ? { page: "report", jobId: decodeURIComponent(parts[1]) } : { page: "video" };
}

export const routeHref = {
  live: "#/",
  video: "#/video",
  reports: "#/reports",
  classic: "#/classic",
  report: (jobId) => `#/video/${encodeURIComponent(jobId)}`,
};

export function useHashRoute() {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return parseRoute(hash);
}
