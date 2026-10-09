# Nashik Kumbh 2027 — Crowd Density & Stampede Risk Prediction

Final-year project: a real-time crowd-safety system for Ramkund Ghat. CSRNet estimates crowd
density from CCTV frames, and Farneback optical flow on short frame bursts measures crowd
turbulence. A crowd-pressure score combines the two into GREEN/YELLOW/ORANGE/RED risk levels.
ORANGE and RED trigger Telegram alerts with a Dijkstra exit route, and a FastAPI +
Socket.IO backend streams live updates to the dashboard.

This is a prototype, demonstrated on recorded festival footage replayed as RTSP. Kumbh's main
bathing days (Aug–Sep 2027) fall after submission, and live deployment would need
police/Mela-authority CCTV access.

## Architecture

```
RTSP cameras / MediaMTX replaying recorded clips
        │  bursts of 5 consecutive frames, every 2 s
        ▼
app.camera_manager ──► Redis list frame_queue:{zone}      (FRAME_SOURCE=local: read in-process)
        │
        ▼
app.worker — every 2 s, per zone
  ├─ CSRNet on the middle frame ──► density map ──► persons/m²   (app.calibration)
  ├─ Farneback flow across the burst ──► turbulence 0–1          (ml.optical_flow)
  └─ RiskEngine: pressure = density × (1 + k·turbulence), 30 s mean ──► GREEN … RED
        ├─► ExitRouter: Dijkstra over live pressure ──► best exits
        ├─► AlertDispatcher: console / Telegram, cooldown + escalation
        └─► MongoDB Atlas: readings, incidents, alerts (saved in the background; falls back to memory)
        ▼
FastAPI + Socket.IO (app.main) ──► React dashboard (frontend/): zone cards, alerts, overlays, trend
```

## Phases

| Phase | Month | What it builds | Status |
|---|---|---|---|
| 1 Foundation | Oct 2026 | Frame extraction, head-point annotator, CVAT import, density maps, infra, alert test | Code done; datasets and annotation in progress |
| 2 AI model | Nov 2026 | CSRNet training/evaluation, optical flow, risk engine, calibration | Code done; training waits for data |
| 3 Backend | Dec 2026 | FastAPI + Socket.IO, inference worker, exit router, Telegram alerts, MongoDB Atlas storage, Docker | Code done; Telegram tested live; MongoDB tested against an in-memory mock, waits for the Atlas connection string |
| 4 Dashboard | Jan 2027 | React dashboard: live zone cards, alerts + RED banner, snapshot with density/Grad-CAM overlays, pressure chart | Code done; checked in a real browser (headless Edge) with a test clip |
| 5 Evaluation | Feb 2027 | Optical flow at 320 px (speed vs accuracy), stress test, public CSRNet weights with provenance, Kumbh footage check, density-weighted turbulence option, training/evaluation scripts, India fine-tuning pipeline; everything in [results/](results/README.md) | Flow benchmark, stress test and Kumbh check measured. Training, the test-set evaluation and the UMN ROC wait for the ShanghaiTech and UMN data; India fine-tuning waits for annotations |
| 6 Thesis | Mar 2027 | Campus calibration test, demo video, thesis | Not started |

## Prerequisites

- Windows 10/11, Python 3.11, Git
- Node.js 22.13 or newer, for the dashboard (Vite 8, Vitest 5)
- NVIDIA GPU with a recent driver (optional; CPU works, slowly)
- A free MongoDB Atlas cluster for storage (optional: without it data is kept in memory); see
  [Database](#database-mongodb-atlas)
- Docker Desktop for Redis and MediaMTX. It needs WSL 2 (`wsl --install` in an
  administrator PowerShell) and CPU virtualisation enabled in the BIOS. Until then, use the
  no-Docker mode below.

## Quick start (Windows, PowerShell)

```powershell
cd "D:\Final Year Claude\nashik-kumbh-stampede-predictor"
python -m venv venv
venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # CUDA build, ~2.6 GB
pip install -r backend\requirements-dev.txt
copy .env.example .env

cd backend
python -m pytest                  # all tests should pass
```

Always run Python with the venv active. The global Python lacks packages such as
`pymongo` and `fakeredis`. All commands below run from `backend\`.

## Running the system

**Demo in one command (no Docker).** From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1                 # alerts as set in .env
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -ConsoleAlerts  # no Telegram messages
powershell -ExecutionPolicy Bypass -File scripts\stop_demo.ps1
```

It starts the backend and the dashboard hidden in the background (logs in `results\demo_*.log`),
waits until the model is loaded, and prints the URL (http://localhost:5173). `-InMemory` keeps
the run out of MongoDB.

**A. Without Docker (by hand).** `FRAME_SOURCE=local` is the default: the backend reads the
cameras and video files itself. Point `CSRNET_WEIGHTS` at a model (for the authors' checkpoint
also set `CSRNET_ALLOW_PICKLE=true`). Put clips in `data\india_dataset\raw_videos\`; each
`rtsp://localhost:8554/<name>` source in `config\zones.json` is then read straight from
`<name>.mp4` (or .avi/.mov/.mkv). A missing clip or an unreachable frame source is logged once,
not once per zone every cycle. Then:

```powershell
uvicorn app.main:app --port 8000          # docs: http://localhost:8000/docs
```

The demo clips (listed with their sources in `data\india_dataset\sources.csv`) are 2025
Prayagraj Maha Kumbh news footage standing in for Nashik cameras; their `area_m2` values are
rough visual estimates, so their densities and risk levels are illustrative only.

Without `MONGODB_URI` (or if Atlas cannot be reached at start-up), readings, incidents and
alerts are kept in memory; `/health` shows `"storage": "memory"` instead of `"mongodb"`.

**B. With Docker (full pipeline).** Set `FRAME_SOURCE=redis` in `.env`.

```powershell
cd ..
docker compose up -d                      # Redis, MediaMTX (the database is Atlas, in the cloud)
cd backend
python -m app.camera_manager              # terminal 1: cameras -> Redis
uvicorn app.main:app --port 8000          # terminal 2: API + inference (GPU)
```

**C. Backend in a container (CPU only):** `docker compose --profile app up -d --build`. The
camera manager still runs on the host.

## Dashboard (frontend)

Vite + React + Tailwind CSS 3 (pinned: v4 has no `npx tailwindcss init`). With the backend running:

```powershell
cd frontend
npm install
copy .env.example .env      # optional: VITE_API_URL, default http://localhost:8000
npm run dev                 # http://localhost:5173
npm test                    # Vitest
npm run build               # production files in frontend\dist
```

To open it on a phone on the same Wi-Fi: start the backend with `--host 0.0.0.0`, set
`VITE_API_URL=http://<laptop IP>:8000` in `frontend\.env`, run `npm run dev -- --host`, and
browse to `http://<laptop IP>:5173`.

What a guard sees:
- **Header:** a Live / Disconnected indicator driven by the Socket.IO connection, the time of
  the last update, and **Enable sound**: once clicked, every new RED alert plays a siren and every
  ORANGE alert a short double beep (Web Audio API, no sound files), each alert once. The button
  then becomes a mute toggle (remembered in this browser).
- **Zone cards:** sorted highest risk first. Each shows a risk badge (icon + word +
  description, never colour alone), people, density, turbulence, crowd pressure, which model
  counted ("YOLO detection" or "CSRNet density map"), the hotspot (most crowded grid cell), the
  **gate actions** for its level, and the top three exits with their paths.
- **No signal:** a zone with no reading for 10 s turns grey and says "No signal"; every zone
  does while disconnected. A no-signal zone ranks above YELLOW, because nobody is watching it.
- **Zone details** (click a card): the camera snapshot every 2 s, an overlay toggle (camera only
  / density heatmap / detections (YOLO person boxes) / Grad-CAM), a 3x3 grid with the people in
  each cell and the hotspot outlined, and crowd pressure for the last 5 minutes with the yellow,
  orange and red thresholds drawn as labelled lines.
- **Alerts:** recent alerts (failed Telegram deliveries are flagged), new ones added live.
  A RED alert shows a full-width banner until dismissed, including a RED raised in the last
  10 minutes before the page was opened.
- **Reports** (header): alert and incident (ORANGE/RED period) history filtered by zone, risk
  level and time range; a per-zone summary (peak people, peak density, peak pressure, time at
  each level); CSV export of each table; **Print** gives a clean printout.

Gate actions (`backend/app/gate_actions.py`) follow from the risk level and the exit graph.
"Gates feeding a zone" are those within 1.5x the nearest gate's walking distance:

| Level | Actions |
|---|---|
| GREEN | Normal operation |
| YELLOW | Monitor the crowd (at the hotspot); prepare marshals at the feeding gates |
| ORANGE | Hold inflow at the nearest feeding gate; open the best other exit (live, congestion-weighted) |
| RED | Stop inflow at all feeding gates; open all exits; announce the route out |

Telegram and dashboard alerts carry the hotspot and these actions.

## Video analysis (uploaded videos)

**Video analysis** in the dashboard header opens an upload page. A recorded crowd video goes
through exactly the live cameras' pipeline, so its numbers compare directly with the live zones:
one burst of 5 consecutive frames every 2 s of *video* time (timestamps from the file, not the
clock), CSRNet on the middle frame, optical-flow turbulence across the burst, crowd pressure,
and a fresh 30 s risk engine.

1. Choose the file (MP4, AVI, MOV or MKV, up to 500 MB). The backend opens it with OpenCV and
   rejects files it cannot decode, whatever their extension.
2. Name the zone and give the **visible area**: the approximate ground area the camera sees
   (e.g. 20 m × 15 m = 300 m², typed directly or as width × length). Density and risk depend on it.
3. Optionally pick its **location on the exit map**, for exit advice. This never changes the
   exit advice for the live zones.
4. **Send Telegram alerts for this video** is off by default. When it is on, every message
   starts with `[VIDEO TEST]`. Dashboard alerts from a video zone always carry that prefix.
5. Upload (with a progress bar), then pick one of:
   - **Analyse once:** the whole video is processed in the background and a report opens.
   - **Add as live camera:** the video loops as a new zone (`VIDEO_1`, ...) on the live
     dashboard, tagged "Test video". Remove it from the Video analysis page.

Shot changes and camera movement are handled the same way as on the live cameras (see
Method notes). A sample whose burst spans a **scene cut** is skipped: it is left out of the
risk average and the summary, shown as a gap in the charts, and flagged `scene cut` in the
CSV. A sample from a **moving camera** (pan, drone, handheld) has the camera's motion removed
before turbulence, is shaded on the charts, and is flagged `moving camera` in the CSV
(`camera_motion_px` gives the estimate).

The report shows the video length, peak count, density and pressure with their times, the
highest risk level, and the time spent at each level. It has a pressure timeline with the
yellow/orange/red lines and a count + turbulence chart. Click a point (or use the slider) to see
that frame with the density heatmap, the YOLO detections or Grad-CAM, plus the 3x3 grid with
its hotspot. The CSV records which model gave each count (`count_method`) and both models'
counts. It also lists the exits at the peak and the
alerts, and offers downloads: the results CSV, and an annotated H.264 MP4 rendered on request
(slow). An orange note says so whenever the loaded model is a demo or untrained model rather
than trained weights.

Analyses run one at a time in a background queue. They work only between live cycles and back
off while a live cycle uses more than 80% of its 2 s budget, so live monitoring stays on time.
Progress arrives over Socket.IO, and the page polls every 2 s if the socket is down. Uploads and
results are kept in `data\uploads\` (git-ignored; `UPLOAD_DIR` and `VIDEO_MAX_MB` in `.env`
change the folder and the size limit). Finished analyses survive a restart. Video zones do not:
add them again after restarting.

To try it before CSRNet is trained, generate the demo model with
`venv\Scripts\python.exe scripts\make_demo.py`. It always predicts the same count, and the
report says so.

## Viva demo script (about 3 minutes)

Before the viva: run `scripts\start_demo.ps1`, open http://localhost:5173 on the projector and
click **Enable sound**. Keep a phone with Telegram on the same Wi-Fi. The three demo clips read
GREEN (their area estimates are large), so to show an escalation add a clip on the Video
analysis page as a live camera with a small visible area (e.g. 20 m²). It is tagged "Test video"
and its alerts start with `[VIDEO TEST]`: say so when presenting.

1. **The live board.** "Each card is one camera zone. The badge carries the level in words,
   not just colour, because guards read this in sunlight." Point at people, density,
   turbulence and pressure.
2. **Escalation.** As the dense clip plays, the zone's pressure rises through YELLOW to ORANGE:
   the card moves to the top and an alert appears with its exit route. At RED, the full-width
   banner appears and the Telegram message arrives on the phone.
3. **Why the system thinks so.** Open the zone. Show the camera, then the density heatmap
   ("where CSRNet counts people"), then Grad-CAM ("which pixels drove the count"). The chart
   shows pressure crossing the dashed thresholds; mention the 30 s smoothing and its delay.
4. **Exit routing.** The recommended exits avoid crowded areas: the cost of leaving an area
   is multiplied by 1 + its pressure.
5. **Failure handling.** Stop the backend: the header turns Disconnected and every card says
   "No signal" instead of showing a stale level. Restart it: everything reconnects by itself.
6. **Honest limits.** A prototype on recorded footage; density needs per-camera calibration;
   the exit graph is schematic; metrics are reported against the paper's MAE 68.2 baseline.

## RTSP test streams (MediaMTX)

Download H.264 clips. YouTube's combined formats now stop at 360p, and AV1 may not decode in
OpenCV:

```powershell
yt-dlp -f "bv*[height<=720][vcodec^=avc1]" -o "..\data\india_dataset\raw_videos\pandharpur_01.%(ext)s" "<YouTube URL>"
```

With `docker compose up -d`, every `data\india_dataset\raw_videos\<name>.mp4` is served in a
loop at `rtsp://localhost:8554/<name>`, with a browser preview at
`http://localhost:8888/<name>`. MediaMTX cannot read a file directly, so [mediamtx.yml](mediamtx.yml)
starts ffmpeg on demand. Without Docker, you can publish a clip by hand (needs
`winget install Gyan.FFmpeg` and `mediamtx.exe` running):

```powershell
ffmpeg -re -stream_loop -1 -i ..\data\india_dataset\raw_videos\pandharpur_01.mp4 -c copy -f rtsp rtsp://localhost:8554/pandharpur_01
```

## Datasets and annotation

Layout and download links: [data/README.md](data/README.md). ShanghaiTech Part A is linked
from https://github.com/leeyeehoo/CSRNet-pytorch, and UCF-QNRF is at
https://www.crcv.ucf.edu/data/ucf-qnrf/.

```powershell
python -m ml.prepare_dataset shanghaitech        # -> data\processed\shanghaitech_a (270 train, 30 val, 182 test)
python -m ml.prepare_dataset qnrf                # long side capped at 2048 px

python -m tools.extract_frames ..\data\india_dataset\raw_videos --fps 0.2           # one frame every 5 s
python -m tools.annotate_points ..\data\india_dataset\frames\pandharpur_01          # click each head; saves as you go
python -m tools.cvat_to_points annotations.xml --video pandharpur_01 --label head   # or import from CVAT
python -m ml.prepare_dataset india --test-videos shirdi_01 --val-videos pandharpur_02   # split by whole video
```

Record every video in `data\india_dataset\sources.csv`; you need these for citations.

## Training and evaluation

```powershell
# RTX 2050 (4 GB): batch 8, 384 px crops, mixed precision
python -m ml.train_csrnet --data ..\data\processed\shanghaitech_a --name parta

# Or, from the project root: the full 200-epoch run in the background, resumable,
# logged to results\train_log.csv (see scripts\train_full.ps1), then the thesis outputs
powershell -ExecutionPolicy Bypass -File scripts\train_full.ps1
venv\Scripts\python.exe scripts\plot_training.py                                         # results\training_curve.png
venv\Scripts\python.exe scripts\report_csrnet.py --weights backend\ml\weights\parta_best.pth   # results\csrnet_*

# Held-out test set: MAE and RMSE -> ml\results\results.csv + heatmaps\
python -m ml.evaluate --weights ml\weights\parta_best.pth --data ..\data\processed\shanghaitech_a --out ml\results
python -m ml.evaluate --weights ml\weights\parta_best.pth --images ..\data\shanghaitech\part_A_final\test_data\images --out ml\results

# Fine-tune on India frames
python -m ml.train_csrnet --data ..\data\processed\india --init ml\weights\parta_best.pth --name india_ft --epochs 60 --lr 1e-6

# Whole pipeline on one clip, offline: annotated video + CSV
python -m tools.analyze_video ..\data\india_dataset\raw_videos\pandharpur_01.mp4 --weights ml\weights\parta_best.pth --area 150
```

The target is the paper's ShanghaiTech Part A result: MAE 68.2, RMSE 115.0. An MAE below 10
is a Part B number. All performance figures here are targets until measured.

India fine-tuning: see [results/INDIA_FINETUNE.md](results/INDIA_FINETUNE.md), which covers CVAT
annotation, the split by video, and `scripts\india_finetune.ps1` for before and after.

## Person detection and the hybrid count

YOLOv8 (ultralytics, COCO-pretrained, person class only, FP16 on the GPU) runs on the same frame
as CSRNet. A detector is accurate when people are few and fully visible; a density map when they
are many and hide each other. The count therefore comes from YOLO when the scene has fewer than
`HYBRID_THRESHOLD` (60) people, otherwise from CSRNet, and every reading records which
(`count_method`: `detection` / `density`) and both counts.

Measured on the three demo clips (640x360, 40 frames each, RTX 2050; `results/yolo_benchmark.md`,
command `venv\Scripts\python.exe scripts\yolo_benchmark.py`):

| Model | Median ms / frame | People found per frame (aerial / ghat / Sangam clip) |
|---|---|---|
| CSRNet (public Part A weights) | 102.1 | 122.0 / 91.3 / 96.3 |
| yolov8n | 13.0 | 0.0 / 10.2 / 5.2 |
| yolov8s (used) | 19.3 | 0.0 / 14.3 / 7.5 |
| yolov8m | 43.8 | 0.0 / 11.4 / 6.6 |

yolov8s finds the most people at 640 px for little extra time; imgsz 1280 found 0.2 per aerial
frame (`results/yolo_benchmark_1280.md`). With CSRNet and YOLO on three zones, `/health` showed a
last cycle of 0.59 s and 0.63 s and a slowest of the last 30 cycles of 0.70 s and 1.09 s in two
runs (2026-10-09), within the 2 s budget.

Because YOLO misses small, distant or hidden people (it finds none in the aerial view of a dense
crowd), its count alone cannot say a scene is sparse: with `HYBRID_RULE=agree` (default) CSRNet
must also count fewer than the threshold. `HYBRID_RULE=detector` is the proposal's rule (YOLO
alone decides). Even with `agree`, in the aerial clip's river shots (t = 66-76 s) CSRNet counted
55-59 and YOLO 0, so the reading became 0: a known limitation for far-off views. Over the
clip's end card both counted near zero (CSRNet 2.5-6.7), where YOLO's 0 is right.

Each camera view is also divided into a `GRID_ROWS` x `GRID_COLS` (3x3) grid. The people in each
cell come from the detected feet when YOLO counted, else from the density map. Persons/m² per
cell use the calibrated area of the cell, or its share of `area_m2`. The densest cell is the
**hotspot**, named in alerts (e.g. "top-right of the view").

## Crowd model and provenance

Every weights file says where it came from, in a `provenance` record inside the checkpoint:
source (`trained_here`, `public_pretrained` or `demo`), dataset, epochs, validation MAE, and the
test MAE/RMSE once measured. Show it with:

```powershell
cd backend
..\venv\Scripts\python.exe -m tools.inspect_weights ml\weights\public\csrnet_partA_public.pth
```

`/health` (`model`, `model_status`), the dashboard footer and the video report name the active
weights.

**No fake numbers.** The backend loads only weights whose provenance says `trained_here` or
`public_pretrained`. Demo weights (`scripts\make_demo.py`) and checkpoints without provenance are
refused unless `DEMO_MODE=1` is set in `.env`. Without valid weights:
- `/health` says `"model_status": "model not loaded"`;
- the dashboard and the Video analysis page say "No trained model: counts unavailable";
- no count, density or risk level is shown, and there is never a fallback number.

Video reports made with demo weights are only served in demo mode. The automated tests use the
demo and fake models.

**Active now:** the CSRNet authors' public ShanghaiTech Part A model
(`CSRNET_WEIGHTS=backend/ml/weights/public/csrnet_partA_public.pth`). It was imported with
`tools.import_public_weights`, which lists what the old pickle would construct (only tensors) and
checks all 34 layer names and shapes. Evaluated once on the 182 Part A test images
(`scripts\report_csrnet.py --record-provenance`): **MAE 71.8, RMSE 120.2** (paper 68.2 / 115.0;
the authors report MAE 66.4). Details are in `results\public_pretrained\`. On the uploaded Kumbh
news clips it undercounts dense and aerial scenes badly
([results/kumbh_check.md](results/kumbh_check.md)), so treat its counts there as relative until it
is fine-tuned on Indian footage.

**Training our own model** (ShanghaiTech Part A, in `data\shanghaitech\`, prepared into
`data\processed\shanghaitech_a`: 270 train, 30 validation from the 300 training images, 182 test):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_full.ps1      # background; survives closing the terminal
powershell -ExecutionPolicy Bypass -File scripts\check_training.ps1  # latest epochs, best val MAE, time left
```

The script trains at batch 8 × 384 px with mixed precision, for up to 400 epochs. It stops early
after 40 epochs without a better validation MAE, and resumes after a crash (or when re-run after
a reboot). Every epoch is logged to `results\train_log.csv`. Training is finished when
`check_training.ps1` shows `Status: done`; the best checkpoint is then
`backend\ml\weights\parta_best.pth`. Evaluate it once:

```powershell
venv\Scripts\python.exe scripts\report_csrnet.py --weights backend\ml\weights\parta_best.pth --record-provenance
```

Switch `CSRNET_WEIGHTS` to it only if its test MAE beats the public weights (71.8).

Rules: evaluate each final model on its test set once (`--record-provenance` refuses a second
time), never choose epochs or settings on the test set, and keep datasets and `.pth` files out of
git (they are ignored).

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, `model_loaded`, `detector`, `hybrid`, zones, storage (mongodb/memory), frame source, camera heartbeats, `last_cycle_s`, `server_time` |
| GET | `/api/config` | Risk thresholds (`yellow`, `orange`, `red`), burst interval, smoothing window, alert cooldown |
| GET | `/api/zones` | Zone configs (camera passwords hidden) |
| GET | `/api/zones/{zone_id}/status` | Latest reading: density, turbulence, pressure, risk_level, count, exits, `server_time` |
| GET | `/api/zones/{zone_id}/history?limit=150` | Last N readings, oldest first (for the trend chart) |
| GET | `/api/zones/{zone_id}/exits?top_k=3` | Best exits under current crowd pressure (Dijkstra) |
| GET | `/api/zones/{zone_id}/snapshot.jpg` | Latest analysed camera frame (at most 960 px wide) |
| GET | `/api/zones/{zone_id}/density.png` | CSRNet density map as a transparent overlay, same size as the snapshot |
| GET | `/api/zones/{zone_id}/detections.png` | YOLO person boxes as a transparent overlay, same size (503 without the detector) |
| GET | `/api/zones/{zone_id}/gradcam.png` | Grad-CAM of the predicted count as a transparent overlay, same size |
| POST | `/api/zones/{zone_id}/infer` | Run inference now on the zone's newest frames |
| GET | `/api/alerts/recent?limit=20` | Recent alerts, newest first, each with its deliveries per channel |
| GET | `/api/incidents?limit=50` | Recent incidents (continuous ORANGE/RED periods) |
| GET | `/api/reports/alerts` (`/incidents`) | Alert (incident) history, newest first; filters `zone_id`, `level` (repeatable), `since`, `until` (epoch s), `limit` |
| GET | `/api/reports/summary` | Per zone: readings, peak count / density / pressure with times, highest level, seconds at each level; filters `zone_id`, `since`, `until` |
| GET | `/api/video/options` | Upload limit, file types, model (trained/demo/untrained), Telegram available, exit-map locations |
| POST | `/api/videos?filename=<name>` | Upload a video as the raw request body; 415 wrong type, 413 too large, 422 not decodable |
| POST | `/api/videos/{video_id}/analyse` | `{name, area_m2, graph_node?, telegram_alerts}` → queued job |
| POST | `/api/videos/{video_id}/live` | Same body → the video loops as a new live zone |
| DELETE | `/api/zones/{zone_id}` | Remove a video zone (409 for a configured camera) |
| GET | `/api/jobs`, `/api/jobs/{job_id}` | Jobs: `queued` / `running` (+ `progress`) / `done` / `failed` (+ `error`) |
| DELETE | `/api/jobs/{job_id}` | Delete a job and its files (stops it if running) |
| GET | `/api/jobs/{job_id}/results` (`.csv`) | Summary, per-sample readings, alerts, exits at the peak (or the readings as CSV) |
| GET | `/api/jobs/{job_id}/samples/{i}/frame.jpg` (`density.png`, `detections.png`, `gradcam.png`) | One analysed frame and its overlays |
| POST / GET | `/api/jobs/{job_id}/annotated` / `annotated.mp4` | Render, then download, the annotated MP4 |

Zone endpoints return 404 for an unknown zone. The image endpoints return 503 with a `detail`
reason when there is no frame or no model yet, and an `X-Frame-Timestamp` header with the
frame's capture time. Times are Unix epoch seconds; `server_time` lets a client judge a
reading's age without trusting its own clock.

Socket.IO (path `/socket.io`) events:
- `density_update` (every 2 s per zone, sent as soon as that zone is processed): `{zone_id,
  name, timestamp, server_time, density, turbulence, pressure, raw_pressure, risk_level, count,
  samples, exits: [{exit, cost, path}], count_method, density_count, detection_count,
  grid: {rows, cols, counts, density, hotspot}, hotspot, gate_actions: [{kind, text, gates}]}`.
  `pressure` is the 30 s mean that sets `risk_level`.
- `alert`: `{id, zone_id, risk_level, message, timestamp, incident_id, delivered,
  deliveries: [{channel, recipient, delivered, latency_ms, error}]}`, the same shape as
  `/api/alerts/recent`.
- `video_job`: a job's state (as `/api/jobs/{job_id}`), at most twice a second while it runs;
  `state: "deleted"` when it is deleted.
- `zones_changed`: `{added}` or `{removed}` when a video zone is added or removed.

## Risk levels

`pressure = density × (1 + 0.5 × turbulence)`, averaged over the last 30 s (15 readings).

| Level | Pressure | Action |
|---|---|---|
| GREEN | < 2.0 | Dashboard only |
| YELLOW | 2.0 – 4.0 | Dashboard warning, security briefed |
| ORANGE | 4.0 – 6.0 | Telegram alert with exit route |
| RED | ≥ 6.0 | All channels, emergency response |

Each zone gets at most one alert per 10 minutes. The exception is an escalation such as
ORANGE → RED, which always fires at once. A zone that stays RED gets a reminder after the
cooldown, not a message every 2 s.

## Alert channels

Set `ALERT_CHANNEL` in `.env` to `console` (default), `telegram`, or `all` (comma-separated
combinations work too). Test it with `python -m tools.send_test_alert`.

| Channel | `.env` keys |
|---|---|
| console | none |
| telegram | `TELEGRAM_BOT_TOKEN` (from @BotFather), `TELEGRAM_CHAT_ID` (send the bot /start once, then read it from `https://api.telegram.org/bot<TOKEN>/getUpdates`) |

SMS through Twilio was removed: India's DLT registration rules block or delay unregistered
SMS, and Telegram needs no registration. To alert several guards, add the bot to a Telegram
group and use the group's chat id (a negative number).

## Database (MongoDB Atlas)

Readings, incidents and alerts go to MongoDB Atlas, database `MONGODB_DB` (default
`kumbhmela`). The fields are listed at the top of [app/store.py](backend/app/store.py).

| Collection | One document per | Index |
|---|---|---|
| `zones` | camera zone (`_id` = zone id; camera passwords hidden) | |
| `zone_readings` | reading: zone, time, pressure, density, turbulence, count, risk level | zone + time |
| `incidents` | continuous ORANGE/RED period, with its peak values | zone + open/closed; start time |
| `alerts` | alert, with its Telegram/console deliveries nested | time |

Set up (free tier, once):

1. Sign up at https://www.mongodb.com/cloud/atlas/register and create a free **M0** cluster in
   the **Mumbai (ap-south-1)** region, the closest one.
2. **Database Access:** add a database user with a password (role "Read and write to any database").
3. **Network Access:** add your current IP address. If you switch networks (college, home,
   phone hotspot), add each one or, for the demo only, allow `0.0.0.0/0`.
4. **Database > Connect > Drivers > Python:** copy the `mongodb+srv://...` string, put the
   password in it, and paste it into `.env` as `MONGODB_URI`. Never commit it.
5. Start the backend: `/health` should say `"storage": "mongodb"`. Collections and indexes are
   created on first use; there are no migrations.

Writes are queued and saved by a background thread, so a slow or dropped connection never
delays the dashboard. While Atlas is unreachable, writes are kept (up to 50,000) and retried
every 2 s, then saved in order. A reading takes about 180 bytes (about 10 MB per camera per day
of non-stop running, with indexes), so the free 512 MB lasts about 12 days of 4 cameras running
24 h; export or delete old `zone_readings` before then. To run the store tests against a real
cluster, set `TEST_MONGODB_URI`; they use, then drop, a `kumbh_pytest` database.

## Repository layout

| Path | Purpose | Plan |
|---|---|---|
| `backend/app/csrnet_model.py` | CSRNet and `DensityEstimator` | §4.1 |
| `backend/ml/optical_flow.py` | Burst Farneback flow, `compute_flow`, turbulence score | §4.2 |
| `backend/app/risk_engine.py` | Crowd pressure, risk levels, `RiskEngine` (rolling mean) | §4.3–4.4 |
| `backend/app/calibration.py` | Floor-marker homography → persons/m² | §4.5 |
| `backend/app/pipeline.py` | One zone, one burst → risk status | Layer 2 |
| `backend/app/camera_manager.py` | Camera reader threads, Redis burst queue, reconnect backoff | Layer 1 |
| `backend/app/exit_router.py` | Walkway graph, Dijkstra exits | Layer 3 |
| `backend/app/alert_dispatcher.py` | Alert policy and channels | Layer 3, §4.6 |
| `backend/app/worker.py`, `store.py`, `main.py` | Inference loop, storage, API | Layer 4 |
| `backend/app/video_analysis.py`, `video_jobs.py`, `video_api.py` | Uploaded videos: live-identical sampling, job queue, video zones, endpoints | Layer 4 |
| `frontend/src/components/VideoPage.jsx`, `VideoReport.jsx` | Video analysis page: upload form, jobs, report | Layer 4 |
| `backend/ml/` | Density maps, dataset prep, training, evaluation | §5, §8 |
| `backend/ml/gradcam.py` | Grad-CAM for CSRNet, hand-written (the pytorch-grad-cam package breaks the annotator's OpenCV) | Layer 4 |
| `backend/tools/` | Frame extraction, annotator, CVAT import, alert test, offline video analysis | Phase 1 |
| `frontend/src/api.js`, `useLiveZones.js` | All network I/O: REST client, Socket.IO, image feed | Layer 4 |
| `frontend/src/risk.js`, `liveState.js` | Risk colours/labels, sort order, 10 s stale rule; live-state reducer | Layer 4 |
| `frontend/src/components/` | Presentational components: cards, detail view, chart, alerts | Layer 4 |
| `config/zones.json` | Cameras, ROIs, areas/calibration, exit-graph nodes | |
| `scripts/` | Phase 5 evaluation: flow benchmark, test clip, VRAM probe, training launcher and curves, CSRNet report, UMN ROC, stress test | §8 |
| `results/` | Tables (CSV + Markdown) and figures (PNG, 200 dpi) for the thesis; see [results/README.md](results/README.md) | §8, Ch. 5–6 |

## Ethics and privacy

- The system processes density maps, not identities: CSRNet outputs a heat map of how many
  people are where, and no face recognition or tracking is done.
- Only aggregate readings (counts, densities, risk levels) are stored. Frames live in Redis
  for seconds and are never written to the database. Camera passwords are removed from the
  zone records before they reach Atlas.
- India-dataset frames from YouTube are kept out of git and must not be published. Release only
  density maps and head-point coordinates, and cite every source video (`sources.csv`).
- The Digital Personal Data Protection Act, 2023 applies to any live CCTV deployment: do a Data
  Protection Impact Assessment and get Mela-authority approval first.
- Camera credentials in RTSP URLs are hidden in API responses.

## Method notes for the thesis

- **Risk score.** `density × (1 + k·turbulence)` is *inspired by* Helbing et al. (2007), whose
  crowd pressure is density × local velocity variance in physical units. Describe it that way.
- **Turbulence** is `1 − |mean v|² / mean |v|²` over the moving pixels of each 32×32 block. It is
  0 for an orderly stream and near 1 for counter-flow. The standard deviation of flow angles
  is not used, because angles wrap at ±180°: a crowd walking left would read as maximum
  turbulence.
- **Camera motion.** For each frame pair an affine model of the whole-frame motion is fitted to
  the flow with RANSAC. When it explains at least half the frame and moves it by at least 1
  px/frame (at 640 px wide), it is subtracted from every pair of the burst before turbulence
  and the sample is marked "moving camera". Slower whole-frame motion is left alone, because a
  dense crowd walking one way across the whole view looks exactly like a slow pan; fixed
  cameras therefore score exactly as before (checked on all 133 bursts of the synthetic clip).
  On synthetic scenes, a pan made a pushing crowd read 0.05 instead of 0.30; after the
  correction it reads 0.29–0.31. Set `"fixed_camera": true` on a zone in `config/zones.json`
  to switch the correction off for a pole-mounted CCTV camera.
- **Scene cuts.** A burst whose frames span a change of shot is skipped (no reading, kept out
  of the 30 s risk average and the video summary) and recorded as "scene cut". A pair is a cut
  when its hue-saturation histogram jumps (Bhattacharyya distance ≥ 0.3; frames within one shot
  of news footage stay below about 0.25, real cuts measured 0.4–0.75) or its grey levels jump
  far more than the burst's other pairs.
- **Density-weighted turbulence (optional).** `TURBULENCE_WEIGHTING=density` weights each flow
  vector by the CSRNet density at that spot, so moving water, flags and sky add nothing. With
  no density map, or a uniform one, it equals the default `frame` score. It stays off until
  `scripts\turbulence_roc.py --compare-density` on UMN shows an AUC at least as high. On the
  Kumbh clips the two agree closely (r 0.89–0.92), but the current density maps miss most of the
  people ([results/kumbh_check.md](results/kumbh_check.md)).
- **Heatmap scale.** Density heatmaps (live overlay, video report, annotated MP4) use one fixed
  persons/m² colour scale tied to the risk thresholds: green at 0, yellow at 2, orange at 4,
  red at 6, dark red from 7. They are never rescaled per frame, so a sparse scene stays faint
  green. Each density cell is converted with the zone's area or calibration. The dashboard and
  the report show the colour key.
- **Flow resolution.** Farneback runs on frames downscaled to `FLOW_MAX_WIDTH` (default 320 px);
  vectors are scaled back to full-resolution pixels, and the window and block sizes (given for
  640 px) scale with the width. On the synthetic test clip this is 4.3× faster than 640 px and
  18× faster than full 1280 px, with r = 0.998 against the full-resolution score
  ([results/flow_speed_vs_accuracy.md](results/flow_speed_vs_accuracy.md)). Re-check on real
  footage.
- **Model selection** uses a validation split taken from the training set, so the test MAE is
  genuinely held out.
- **Alert delay.** The 30 s rolling mean can add up to ~30 s before an alert. Report it
  separately from delivery latency (§4.6).
- **Exit routing.** Pressure in an area multiplies the cost of leaving it by `1 + pressure`, so
  routes avoid RED areas. The graph's layout and distances are placeholders until measured on
  the site map.

## References

1. Y. Li, X. Zhang, D. Chen, "CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes," *CVPR*, 2018.
2. Y. Zhang, D. Zhou, S. Chen, S. Gao, Y. Ma, "Single-Image Crowd Counting via Multi-Column Convolutional Neural Network," *CVPR*, 2016 (ShanghaiTech dataset).
3. H. Idrees et al., "Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds," *ECCV*, 2018 (UCF-QNRF dataset).
4. D. Helbing, A. Johansson, H. Z. Al-Abideen, "Dynamics of crowd disasters: An empirical study," *Physical Review E*, 75, 046109, 2007.
5. G. Farnebäck, "Two-Frame Motion Estimation Based on Polynomial Expansion," *SCIA*, 2003.
6. J. J. Fruin, *Pedestrian Planning and Design*, 1971.
7. The Digital Personal Data Protection Act, 2023 (India).
