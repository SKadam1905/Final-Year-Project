"""Endpoints for uploaded videos (the dashboard's "Video analysis" page).

    GET    /api/video/options                     upload limits, model, Telegram, exit-map locations
    POST   /api/videos?filename=<name>            upload: the file is the raw request body (streamed)
    POST   /api/videos/{video_id}/analyse         {name, area_m2, graph_node?, telegram_alerts} -> queued job
    POST   /api/videos/{video_id}/live            same body -> the video loops as a new live zone
    DELETE /api/zones/{zone_id}                   remove a video zone (configured cameras cannot be removed)
    GET    /api/jobs                              all jobs, newest first
    GET    /api/jobs/{job_id}                     state (queued / running + progress / done / failed + error)
    DELETE /api/jobs/{job_id}                     delete a job and its files (stops it if running)
    GET    /api/jobs/{job_id}/results             summary, per-sample readings, alerts, exits at the peak
    GET    /api/jobs/{job_id}/results.csv         the readings as CSV
    GET    /api/jobs/{job_id}/samples/{i}/frame.jpg | density.png | detections.png | gradcam.png
    POST   /api/jobs/{job_id}/annotated           start rendering the annotated MP4 (slow)
    GET    /api/jobs/{job_id}/annotated.mp4       the annotated MP4 once rendered
Socket.IO: "video_job" (a job's state, at most twice a second while it runs) and
"zones_changed" ({added} or {removed}) when a video zone is added or removed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.video_analysis import VideoRejected, check_extension, safe_filename
from app.video_jobs import VideoService, VideoZoneRequest
from app.calibration import ZoneGeometry
from app.visualize import colour_overlay, density_rgba, density_scale, detections_rgba

GRADCAM_WIDTH = 640


class VideoZoneBody(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    area_m2: float = Field(gt=0, le=1_000_000, description="ground area the camera sees, m²")
    graph_node: str | None = Field(None, description="location on the exit map, for exit advice")
    telegram_alerts: bool = False


def add_video_routes(api: FastAPI, video: Callable[[], VideoService]) -> None:
    def job_or_404(job_id: str):
        job = video().jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"no video job {job_id!r}")
        return job

    def done_job(job_id: str):
        job = job_or_404(job_id)
        if job.state != "done":
            raise HTTPException(409, f"the analysis is {job.state}, not done")
        return job

    def request_for(body: VideoZoneBody) -> VideoZoneRequest:
        v = video()
        options = v.options()
        if body.telegram_alerts and not options["telegram_available"]:
            raise HTTPException(400, "Telegram is not set up: add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env")
        if body.graph_node and body.graph_node not in options["locations"]:
            raise HTTPException(422, f"unknown exit-map location {body.graph_node!r}")
        if not options["model_loaded"]:
            raise HTTPException(503, f"no crowd model loaded: {options['model_error']}")
        return VideoZoneRequest(body.name.strip(), body.area_m2, body.graph_node or None, body.telegram_alerts)

    def video_or_404(video_id: str) -> dict:
        try:
            return video().video(video_id)
        except KeyError:
            raise HTTPException(404, f"no uploaded video {video_id!r}; upload it again") from None

    @api.get("/api/video/options")
    def video_options():
        return video().options()

    @api.post("/api/videos", status_code=201)
    async def upload_video(request: Request, filename: str):
        """Stream the body to disk, stopping as soon as it passes the size limit, then make sure
        OpenCV can decode it (an extension alone proves nothing)."""
        v = video()
        limit = v.settings.video_max_mb * 1024 * 1024
        too_large = HTTPException(413, f"The file is larger than the {v.settings.video_max_mb} MB limit.")
        try:
            check_extension(filename)
        except VideoRejected as exc:
            raise HTTPException(exc.status, str(exc)) from None
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise too_large
        video_id, path = v.new_upload(safe_filename(filename))
        size = 0
        try:
            with open(path, "wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise too_large
                    f.write(chunk)
            if size == 0:
                raise HTTPException(422, "The upload was empty.")
            return await asyncio.to_thread(v.register_upload, video_id, path, Path(filename).name, size)
        except VideoRejected as exc:
            raise HTTPException(exc.status, str(exc)) from None
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    @api.post("/api/videos/{video_id}/analyse", status_code=202)
    def analyse(video_id: str, body: VideoZoneBody):
        video_or_404(video_id)
        job = video().create_job(video_id, request_for(body))
        return job.public()

    @api.post("/api/videos/{video_id}/live", status_code=201)
    async def add_live(video_id: str, body: VideoZoneBody):
        video_or_404(video_id)
        request = request_for(body)
        try:
            zone = video().add_live(video_id, request)  # on the event loop: no cycle is being scheduled
        except OverflowError as exc:
            raise HTTPException(409, str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from None
        return {"id": zone.id, "name": zone.name, "area_m2": zone.area_m2, "kind": "video",
                "graph_node": request.graph_node, "telegram_alerts": request.telegram_alerts}

    @api.delete("/api/zones/{zone_id}", status_code=204)
    async def remove_zone(zone_id: str):
        v = video()
        if zone_id not in v.video_zones:
            if zone_id in v.services.zones:
                raise HTTPException(409, f"{zone_id!r} is a configured camera; only video zones can be removed here")
            raise HTTPException(404, f"unknown zone {zone_id!r}")
        await asyncio.to_thread(v.remove_live, zone_id)  # waits for a step already running on it
        return Response(status_code=204)

    @api.get("/api/jobs")
    def list_jobs():
        return video().list_jobs()

    @api.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = job_or_404(job_id)
        return {**job.public(), "queue_position": video().queue_position(job_id) if job.state == "queued" else 0}

    @api.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str):
        job_or_404(job_id)
        video().delete_job(job_id)
        return Response(status_code=204)

    def trusted_results(job_id: str) -> dict:
        """A finished job's results, unless they were made with demo / provenance-less weights:
        those counts are not real, so they are only served with DEMO_MODE=1."""
        from app.csrnet_model import TRUSTED_KINDS

        done_job(job_id)
        results = video().results(job_id)
        model = results.get("model") or {}
        if model.get("kind") not in TRUSTED_KINDS and not video().settings.demo_mode:
            raise HTTPException(409, f"No trained model: counts unavailable. This analysis was made with "
                                     f"{model.get('kind', 'unknown')} weights ({model.get('weights')}), so its counts are "
                                     "not real. Delete it and analyse the video again with trained weights.")
        return results

    @api.get("/api/jobs/{job_id}/results")
    def job_results(job_id: str):
        return trusted_results(job_id)

    @api.get("/api/jobs/{job_id}/results.csv")
    def job_csv(job_id: str):
        job = done_job(job_id)
        trusted_results(job_id)
        name = f"{safe_filename(job.name).rsplit('.', 1)[0]}_{job_id}_readings.csv"
        return FileResponse(video().job_dir(job_id) / "results.csv", media_type="text/csv", filename=name)

    def sample_frame(job_id: str, index: int) -> Path:
        done_job(job_id)
        path = video().job_dir(job_id) / "frames" / f"{index:05d}.jpg"
        if not path.exists():
            raise HTTPException(404, f"no sample {index}")
        return path

    def png(image: np.ndarray) -> Response:
        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            raise HTTPException(500, "image encoding failed")
        return Response(buffer.tobytes(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @api.get("/api/jobs/{job_id}/samples/{index}/frame.jpg")
    def job_frame(job_id: str, index: int):
        return FileResponse(sample_frame(job_id, index), media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=3600"})

    @api.get("/api/jobs/{job_id}/samples/{index}/density.png")
    def job_density(job_id: str, index: int):
        """persons/m² on the same fixed scale as the live heatmap (404 for a scene cut: no density)."""
        job = done_job(job_id)
        frame = cv2.imread(str(sample_frame(job_id, index)))
        path = video().job_dir(job_id) / "density" / f"{index:05d}.npy"
        if not path.exists():
            raise HTTPException(404, f"sample {index} was not analysed (scene cut)")
        size = (frame.shape[1], frame.shape[0])
        per_m2 = ZoneGeometry(area_m2=job.area_m2).cell_density(np.load(path).astype(np.float32), size)
        s = video().settings
        return png(density_rgba(per_m2, size, density_scale(s.risk_yellow, s.risk_orange, s.risk_red)))

    @api.get("/api/jobs/{job_id}/samples/{index}/detections.png")
    def job_detections(job_id: str, index: int):
        """YOLO person boxes (404 for a scene cut, or a job analysed without the detector)."""
        frame = cv2.imread(str(sample_frame(job_id, index)))
        path = video().job_dir(job_id) / "detections" / f"{index:05d}.npy"
        if not path.exists():
            raise HTTPException(404, f"sample {index} has no person detections")
        size = (frame.shape[1], frame.shape[0])
        return png(detections_rgba(np.load(path), size, size))

    @api.get("/api/jobs/{job_id}/samples/{index}/gradcam.png")
    def job_gradcam(job_id: str, index: int):
        frame_path = sample_frame(job_id, index)
        cached = video().job_dir(job_id) / "gradcam" / f"{index:05d}.png"
        if not cached.exists():
            gradcam = video().services.gradcam
            if gradcam is None:
                raise HTTPException(503, "Grad-CAM needs the crowd model, which is not loaded")
            from app.csrnet_model import to_input

            frame = cv2.imread(str(frame_path))
            cam = gradcam(to_input(frame, GRADCAM_WIDTH).to(gradcam.device))
            cached.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(cached), colour_overlay(cam, (frame.shape[1], frame.shape[0])))
        return FileResponse(cached, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @api.post("/api/jobs/{job_id}/annotated", status_code=202)
    def start_annotated(job_id: str):
        done_job(job_id)
        return video().request_annotated(job_id).public()

    @api.get("/api/jobs/{job_id}/annotated.mp4")
    def annotated_video(job_id: str):
        job = done_job(job_id)
        trusted_results(job_id)
        if job.annotated["state"] != "done":
            raise HTTPException(409, f"the annotated video is {job.annotated['state']}; request it first")
        name = f"{safe_filename(job.name).rsplit('.', 1)[0]}_{job_id}_annotated.mp4"
        return FileResponse(video().job_dir(job_id) / "annotated.mp4", media_type="video/mp4", filename=name)
