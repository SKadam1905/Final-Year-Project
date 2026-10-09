import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import VideoUploadForm from "./VideoUploadForm.jsx";

const OPTIONS = { max_upload_mb: 500, telegram_available: true, model_loaded: true, locations: ["GHAT_1", "RAMKUND_NORTH"] };
const VIDEO = { id: "v1", filename: "ghat.mp4", duration_s: 75, width: 1280, height: 720, fps: 25 };

const videoFile = (name = "ghat.mp4", bytes = 1000) => new File([new Uint8Array(bytes)], name, { type: "video/mp4" });

function setup(props = {}) {
  const user = userEvent.setup({ applyAccept: false });
  const handlers = { onAnalyse: vi.fn().mockResolvedValue(undefined), onAddLive: vi.fn().mockResolvedValue(undefined) };
  render(<VideoUploadForm options={OPTIONS} {...handlers} {...props} />);
  return { user, ...handlers };
}

describe("VideoUploadForm", () => {
  it("explains what is missing instead of uploading", async () => {
    const upload = vi.fn();
    const { user } = setup({ upload });
    await user.click(screen.getByRole("button", { name: "Upload video" }));
    expect(screen.getByText("Choose a video file.")).toBeInTheDocument();
    expect(screen.getByText("Give the zone a name.")).toBeInTheDocument();
    expect(screen.getByText("Enter the visible area in square metres (above 0).")).toBeInTheDocument();
    expect(upload).not.toHaveBeenCalled();
  });

  it("rejects a file that is not a video as soon as it is chosen", async () => {
    const { user } = setup();
    await user.upload(screen.getByLabelText("Video file"), new File(["x"], "notes.txt", { type: "text/plain" }));
    expect(screen.getByText(/Unsupported file type/)).toBeInTheDocument();
  });

  it("shows the area help, works out width x length, uploads with progress, then offers both modes", async () => {
    let finish;
    const upload = vi.fn((file, onProgress) => {
      onProgress(40);
      return new Promise((resolve) => {
        finish = () => resolve(VIDEO);
      });
    });
    const { user, onAnalyse, onAddLive } = setup({ upload });
    expect(screen.getByText(/e\.g\. 20 m x 15 m = 300 m²\. Density and risk depend on this\./)).toBeInTheDocument();

    await user.upload(screen.getByLabelText("Video file"), videoFile());
    expect(screen.getByLabelText("Zone name")).toHaveValue("ghat"); // filled from the file name
    await user.click(screen.getByLabelText("Width × length in metres"));
    await user.type(screen.getByLabelText("Width (m)"), "20");
    await user.type(screen.getByLabelText("Length (m)"), "15");
    expect(screen.getByText("= 300 m²")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText(/Location on the exit map/), "GHAT_1");
    expect(screen.getByLabelText("Send Telegram alerts for this video")).not.toBeChecked(); // off by default

    await user.click(screen.getByRole("button", { name: "Upload video" }));
    expect(screen.getByRole("progressbar", { name: "Uploading" })).toHaveAttribute("aria-valuenow", "40");
    finish();
    expect(await screen.findByText(/Uploaded: ghat\.mp4 · 1:15 · 1280×720 · 25 fps/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Analyse once" }));
    expect(onAnalyse).toHaveBeenCalledWith(VIDEO, { name: "ghat", area_m2: 300, graph_node: "GHAT_1", telegram_alerts: false });
    await user.click(screen.getByLabelText("Send Telegram alerts for this video"));
    await user.click(screen.getByRole("button", { name: "Add as live camera" }));
    expect(onAddLive).toHaveBeenCalledWith(VIDEO, { name: "ghat", area_m2: 300, graph_node: "GHAT_1", telegram_alerts: true });
  });

  it("shows the backend's reason when it rejects the file", async () => {
    const upload = vi.fn().mockRejectedValue(new Error("This file could not be opened as a video."));
    const { user } = setup({ upload });
    await user.upload(screen.getByLabelText("Video file"), videoFile("broken.mp4"));
    await user.type(screen.getByLabelText("Area (m²)"), "300");
    await user.click(screen.getByRole("button", { name: "Upload video" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("This file could not be opened as a video.");
    expect(screen.queryByRole("button", { name: "Analyse once" })).not.toBeInTheDocument();
  });

  it("offers no analysis without a trained model", async () => {
    const upload = vi.fn().mockResolvedValue(VIDEO);
    const { user, onAnalyse } = setup({ upload, options: { ...OPTIONS, model_loaded: false } });
    await user.upload(screen.getByLabelText("Video file"), videoFile());
    await user.type(screen.getByLabelText("Area (m²)"), "300");
    await user.click(screen.getByRole("button", { name: "Upload video" }));
    expect(await screen.findByRole("button", { name: "Analyse once" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Add as live camera" })).toBeDisabled();
    expect(onAnalyse).not.toHaveBeenCalled();
  });

  it("disables Telegram when the backend has no bot configured", () => {
    setup({ options: { ...OPTIONS, telegram_available: false } });
    expect(screen.getByLabelText("Send Telegram alerts for this video")).toBeDisabled();
    expect(screen.getByText(/Telegram is not set up/)).toBeInTheDocument();
  });
});
