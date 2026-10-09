import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import GridOverlay, { GridCaption } from "./GridOverlay.jsx";

const GRID = {
  rows: 3, cols: 3,
  counts: [[1, 2, 24], [0, 3, 0], [0, 0, 1]],
  density: [[0.1, 0.2, 2.4], [0, 0.3, 0], [0, 0, 0.1]],
  hotspot: { row: 0, col: 2, label: "top-right", count: 24, density: 2.4 },
};

describe("GridOverlay", () => {
  it("draws one cell per grid square with its count and outlines the hotspot", () => {
    render(<GridOverlay grid={GRID} />);
    const overlay = screen.getByTestId("grid-overlay");
    expect(overlay.children).toHaveLength(9);
    expect(overlay.children[2]).toHaveTextContent("Hotspot · 24");
    expect(overlay.children[2]).toHaveClass("border-red-600");
    expect(overlay.children[4]).toHaveTextContent("3");
  });

  it("describes the hotspot in text for screen readers", () => {
    render(<GridCaption grid={GRID} />);
    expect(screen.getByText(/3 × 3 cells\. Hotspot: top-right of the view: ~24 people \(2\.40 \/m²\)/)).toBeInTheDocument();
  });

  it("draws nothing without a grid (readings from before zone-wise segmentation)", () => {
    const { container } = render(<GridOverlay grid={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
