import { describe, expect, it } from "vitest";
import { parseRoute } from "./useHashRoute.js";
import { applyJob, isWorking } from "./useVideoJobs.js";
import { areaFrom, emptyVideoForm, fileError, nameFromFile, zoneErrors, zoneRequest } from "./videoForm.js";

const file = (name, mb = 10) => ({ name, size: mb * 1024 * 1024 });

describe("video upload form rules", () => {
  it("accepts only video types under the size limit", () => {
    expect(fileError(null, 500)).toBe("Choose a video file.");
    expect(fileError(file("notes.txt"), 500)).toMatch(/Unsupported file type/);
    expect(fileError(file("clip.MOV"), 500)).toBeNull();
    expect(fileError(file("big.mp4", 501), 500)).toBe("The file is 501 MB; the limit is 500 MB.");
    expect(fileError(file("empty.mkv", 0), 500)).toBe("The file is empty.");
  });

  it("takes the area directly or as width x length", () => {
    expect(areaFrom({ ...emptyVideoForm, area: "300" })).toBe(300);
    expect(areaFrom({ ...emptyVideoForm, area: "0" })).toBeNull();
    expect(areaFrom({ ...emptyVideoForm, areaMode: "size", width: "20", length: "15" })).toBe(300);
    expect(areaFrom({ ...emptyVideoForm, areaMode: "size", width: "20", length: "" })).toBeNull();
    expect(areaFrom({ ...emptyVideoForm, area: "12,5" })).toBe(12.5);
  });

  it("requires a name and an area", () => {
    expect(zoneErrors(emptyVideoForm)).toEqual({
      name: "Give the zone a name.",
      area: "Enter the visible area in square metres (above 0).",
    });
    expect(zoneErrors({ ...emptyVideoForm, name: "Ghat", areaMode: "size", width: "20" }).area).toMatch(/width and the length/);
    expect(zoneErrors({ ...emptyVideoForm, name: "Ghat", area: "2000000" }).area).toMatch(/check the numbers/);
    expect(zoneErrors({ ...emptyVideoForm, name: "  Ghat ", area: "300" })).toEqual({});
  });

  it("builds the request body", () => {
    const form = { ...emptyVideoForm, name: " Ghat 1 ", areaMode: "size", width: "20", length: "15", location: "GHAT_1", telegram: true };
    expect(zoneRequest(form)).toEqual({ name: "Ghat 1", area_m2: 300, graph_node: "GHAT_1", telegram_alerts: true });
    expect(zoneRequest({ ...emptyVideoForm, name: "x", area: "50" }).graph_node).toBeNull();
    expect(nameFromFile("Ramkund evening.mp4")).toBe("Ramkund evening");
  });
});

describe("video pages", () => {
  it("routes by hash", () => {
    expect(parseRoute("")).toEqual({ page: "live" });
    expect(parseRoute("#/video")).toEqual({ page: "video" });
    expect(parseRoute("#/video/abc123")).toEqual({ page: "report", jobId: "abc123" });
  });

  it("merges job events and drops deleted jobs", () => {
    let jobs = applyJob({}, { id: "a", state: "queued", progress: 0, annotated: { state: "none" } });
    expect(isWorking(jobs.a)).toBe(true);
    jobs = applyJob(jobs, { id: "a", state: "done", progress: 100, annotated: { state: "none" } });
    expect(isWorking(jobs.a)).toBe(false);
    expect(applyJob(jobs, { id: "a", state: "deleted" })).toEqual({});
  });
});
