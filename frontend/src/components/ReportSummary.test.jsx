import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import DensityLegend from "./DensityLegend.jsx";
import ModelNotice from "./ModelNotice.jsx";
import ReportSummary from "./ReportSummary.jsx";

const SUMMARY = {
  duration_s: 75.4, samples: 38,
  peak_count: 1412.6, peak_count_time_s: 34.08,
  peak_density: 7.06, peak_density_time_s: 34.08,
  peak_pressure: 6.42, peak_pressure_time_s: 40.08, peak_pressure_sample: 20,
  peak_raw_pressure: 7.9, highest_level: "RED",
  time_at_level_s: { GREEN: 20, YELLOW: 25.4, ORANGE: 18, RED: 12 },
};

describe("ReportSummary", () => {
  it("shows the length, peaks with their times, the highest level and time at each level", () => {
    render(<ReportSummary summary={SUMMARY} />);
    const tile = (label) => screen.getByText(label).parentElement;
    expect(within(tile("Video length")).getByText("1:15")).toBeInTheDocument();
    expect(within(tile("Peak count")).getByText("1,413 people")).toBeInTheDocument();
    expect(within(tile("Peak count")).getByText("at 0:34")).toBeInTheDocument();
    expect(within(tile("Peak density")).getByText("7.06 /m²")).toBeInTheDocument();
    expect(within(tile("Peak crowd pressure")).getByText("6.42")).toBeInTheDocument();
    expect(within(tile("Peak crowd pressure")).getByText("30 s mean, at 0:40")).toBeInTheDocument();
    expect(within(tile("Highest risk level")).getByText("RED")).toBeInTheDocument();

    const levels = screen.getByText("Time at each risk level").nextElementSibling.nextElementSibling;
    expect(within(levels).getAllByRole("listitem").map((li) => li.textContent)).toEqual([
      "■ RED: 0:12 (16%)", "◆ ORANGE: 0:18 (24%)", "▲ YELLOW: 0:25 (34%)", "● GREEN: 0:20 (27%)",
    ]);
    expect(screen.queryByText(/Scene cuts:/)).not.toBeInTheDocument(); // none in this video
  });

  it("reports skipped scene cuts and moving-camera samples", () => {
    render(<ReportSummary summary={{ ...SUMMARY, scene_cut_samples: 4, time_scene_cut_s: 8, moving_camera_samples: 9 }} />);
    expect(screen.getByText(/Scene cuts:/).parentElement).toHaveTextContent(
      "Scene cuts: 4 samples spanned a change of shot (0:08 of video). They were skipped and are not in the numbers above.");
    expect(screen.getByText(/Moving camera:/).parentElement).toHaveTextContent("Moving camera: 9 samples.");
  });
});

describe("DensityLegend", () => {
  const SCALE = { unit: "persons/m²", max: 7, stops: [
    { value: 0, colour: "#15803d" }, { value: 2, colour: "#facc15" }, { value: 4, colour: "#f97316" },
    { value: 6, colour: "#b91c1c" }, { value: 7, colour: "#7f1d1d" }] };

  it("shows the fixed persons/m² scale with its ticks", () => {
    const { container } = render(<DensityLegend scale={SCALE} />);
    expect(screen.getByRole("figure", { name: "Heatmap colour key in persons/m²" })).toBeInTheDocument();
    expect(["0", "2", "4", "6", "7+"].every((t) => screen.getByText(t))).toBe(true);
    expect(container.querySelector("[style*=linear-gradient]").getAttribute("style"))
      .toContain("rgb(250, 204, 21) 28.5"); // yellow at 2 of 7
    expect(screen.getByText(/yellow from 2, orange from 4, red from 6/)).toBeInTheDocument();
  });
});

describe("ModelNotice", () => {
  it("names the weights in use and warns for demo weights", () => {
    const { rerender } = render(<ModelNotice model={{ kind: "demo", label: "Demo model", weights: "demo_csrnet.pth" }} />);
    expect(screen.getByRole("note")).toHaveTextContent("Demo model: counts are not real estimates");
    rerender(<ModelNotice model={{ kind: "public_pretrained", label: "Public pretrained CSRNet (ShanghaiTech Part A)" }} />);
    expect(screen.getByRole("note")).toHaveTextContent("Model: public pretrained CSRNet");
    expect(screen.getByRole("note")).not.toHaveTextContent("not real");
    rerender(<ModelNotice model={null} />);
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
  });
});
