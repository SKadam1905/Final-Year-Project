// Video-analysis jobs: the list from GET /api/jobs, kept current by the "video_job" Socket.IO
// event. While the socket is down and a job is still working, the list is polled every 2 s
// instead, so progress keeps moving either way.

import { useCallback, useEffect, useState } from "react";
import { io } from "socket.io-client";
import { API_URL, socketOptions, socketRefused, videoApi } from "./api.js";

const POLL_MS = 2000;

export function isWorking(job) {
  return ["queued", "running"].includes(job.state) || ["queued", "running"].includes(job.annotated?.state);
}

/** A "video_job" event (or a REST job) merged into the jobs map; state "deleted" removes it. */
export function applyJob(jobs, job) {
  const next = { ...jobs };
  if (job.state === "deleted") delete next[job.id];
  else next[job.id] = { ...next[job.id], ...job };
  return next;
}

/** `enabled` false (a Security Officer: jobs are admin-only) makes no requests at all. */
export function useVideoJobs(enabled = true) {
  const [jobs, setJobs] = useState({});
  const [connected, setConnected] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const list = await videoApi.jobs();
      setJobs(Object.fromEntries(list.map((job) => [job.id, job])));
      setLoaded(true);
    } catch {
      // backend unreachable: keep what we have; the live page shows the connection problem
    }
  }, []);

  useEffect(() => {
    if (!enabled) return undefined;
    refresh();
    const socket = io(API_URL, socketOptions({ reconnectionDelayMax: 5000 }));
    const connectTimer = setTimeout(() => socket.connect(), 0); // see useLiveZones.js
    socket.on("connect", () => {
      setConnected(true);
      refresh(); // catch up on anything missed while disconnected
    });
    socket.on("disconnect", () => setConnected(false));
    socket.on("connect_error", (error) => {
      setConnected(false);
      if (socketRefused(error)) socket.disconnect();
    });
    socket.on("video_job", (job) => setJobs((previous) => applyJob(previous, job)));
    return () => {
      clearTimeout(connectTimer);
      socket.disconnect();
    };
  }, [refresh, enabled]);

  const working = Object.values(jobs).some(isWorking);
  useEffect(() => {
    if (connected || !working) return undefined;
    const timer = setInterval(refresh, POLL_MS);
    return () => clearInterval(timer);
  }, [connected, working, refresh]);

  const upsert = useCallback((job) => setJobs((previous) => applyJob(previous, job)), []);
  const list = Object.values(jobs).sort((a, b) => b.created_at - a.created_at);
  return { jobs, list, loaded, connected, refresh, upsert };
}
