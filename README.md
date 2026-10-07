# Nashik Kumbh 2027 — Crowd Density & Stampede Risk Prediction

Final-year project: a real-time crowd-safety system for Ramkund Ghat. CSRNet estimates crowd
density from CCTV frames, and Farneback optical flow on short frame bursts measures crowd
turbulence. A crowd-pressure score combines the two into GREEN/YELLOW/ORANGE/RED risk levels.
ORANGE and RED trigger SMS/Telegram alerts with a Dijkstra exit route, and a FastAPI +
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
        ├─► AlertDispatcher: console / SMS / Telegram, cooldown + escalation ──► AlertLog
        └─► PostgreSQL + PostGIS: readings, incidents, alerts (falls back to memory)
        ▼
FastAPI + Socket.IO (app.main) ──► React + Leaflet dashboard (Phase 4)
```

## Phases

| Phase | Month | What it builds | Status |
|---|---|---|---|
| 1 Foundation | Oct 2026 | Frame extraction, head-point annotator, CVAT import, density maps, infra, alert test | Code done; datasets and annotation in progress |
| 2 AI model | Nov 2026 | CSRNet training/evaluation, optical flow, risk engine, calibration | Code done; training waits for data |
| 3 Backend | Dec 2026 | FastAPI + Socket.IO, inference worker, exit router, alerts, DB + migrations, Docker | Code done; not yet run against a live PostGIS |
| 4 Dashboard | Jan 2027 | React + Leaflet + Recharts, Grad-CAM overlay | Not started |
| 5–6 Evaluation, thesis | Feb–Mar 2027 | Campus calibration test, UMN turbulence ROC, demo video, thesis | Not started |

## Prerequisites

- Windows 10/11, Python 3.11, Git
- NVIDIA GPU with a recent driver (optional; CPU works, slowly)
- Docker Desktop for PostGIS, Redis and MediaMTX. It needs WSL 2 (`wsl --install` in an
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
`geoalchemy2` and `fakeredis`. All commands below run from `backend\`.

## Running the system

**A. Without Docker (works today).** In `.env` set `FRAME_SOURCE=local` and point
`CSRNET_WEIGHTS` at a model (for the authors' checkpoint also set `CSRNET_ALLOW_PICKLE=true`).
Put clips in `data\india_dataset\raw_videos\`; each `rtsp://localhost:8554/<name>` source in
`config\zones.json` is then read straight from `<name>.mp4`. Then:

```powershell
uvicorn app.main:app --port 8000          # docs: http://localhost:8000/docs
```

Without a database, readings, incidents and alerts are kept in memory; `/health` shows
`"storage": "memory"`.

**B. With Docker (full pipeline).**

```powershell
cd ..
docker compose up -d                      # PostGIS, Redis, MediaMTX
cd backend
alembic upgrade head                      # create the tables (once)
python -m app.camera_manager              # terminal 1: cameras -> Redis
uvicorn app.main:app --port 8000          # terminal 2: API + inference (GPU)
```

**C. Backend in a container (CPU only):** `docker compose --profile app up -d --build`, then
`docker compose run --rm backend alembic upgrade head`. The camera manager still runs on the
host.

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
# RTX 2050 (4 GB): batch 8, 384 px crops, mixed precision, about 1 min per epoch
python -m ml.train_csrnet --data ..\data\processed\shanghaitech_a --name parta

# Held-out test set: MAE and RMSE -> ml\results\results.csv + heatmaps\
python -m ml.evaluate --weights ml\weights\parta_best.pth --data ..\data\processed\shanghaitech_a --out ml\results
python -m ml.evaluate --weights ml\weights\parta_best.pth --images ..\data\shanghaitech_part_a\part_A_final\test_data\images --out ml\results

# Fine-tune on India frames
python -m ml.train_csrnet --data ..\data\processed\india --init ml\weights\parta_best.pth --name india_ft --epochs 60 --lr 1e-6

# Whole pipeline on one clip, offline: annotated video + CSV
python -m tools.analyze_video ..\data\india_dataset\raw_videos\pandharpur_01.mp4 --weights ml\weights\parta_best.pth --area 150
```

The target is the paper's ShanghaiTech Part A result: MAE 68.2, RMSE 115.0. An MAE below 10
is a Part B number. All performance figures here are targets until measured.

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, `model_loaded`, zones, storage (postgres/memory), frame source, camera heartbeats |
| GET | `/api/zones` | Zone configs (camera passwords hidden) |
| GET | `/api/zones/{zone_id}/status` | Latest reading: density, turbulence, pressure, risk_level, count, exits |
| GET | `/api/zones/{zone_id}/exits?top_k=3` | Best exits under current crowd pressure (Dijkstra) |
| POST | `/api/zones/{zone_id}/infer` | Run inference now on the zone's newest frames |
| GET | `/api/alerts/recent?limit=20` | Recent alerts, one row per channel and recipient |
| GET | `/api/incidents?limit=50` | Recent incidents (continuous ORANGE/RED periods) |

Socket.IO (path `/socket.io`) events:
- `density_update` (every 2 s per zone): `{zone_id, name, density, turbulence, pressure,
  raw_pressure, risk_level, count, timestamp, exits}`
- `alert`: `{zone_id, risk_level, message, exits, timestamp, delivered}`

## Risk levels

`pressure = density × (1 + 0.5 × turbulence)`, averaged over the last 30 s (15 readings).

| Level | Pressure | Action |
|---|---|---|
| GREEN | < 2.0 | Dashboard only |
| YELLOW | 2.0 – 4.0 | Dashboard warning, security briefed |
| ORANGE | 4.0 – 6.0 | SMS/Telegram alert with exit route |
| RED | ≥ 6.0 | All channels, emergency response |

Each zone gets at most one alert per 10 minutes. The exception is an escalation such as
ORANGE → RED, which always fires at once. A zone that stays RED gets a reminder after the
cooldown, not an SMS every 2 s.

## Alert channels

Set `ALERT_CHANNEL` in `.env` to `console` (default), `sms` (or `twilio`), `telegram`, or `all`.
Comma-separated combinations work too. Test them with `python -m tools.send_test_alert`.

| Channel | `.env` keys |
|---|---|
| console | none |
| sms (Twilio) | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `ALERT_SMS_TO` (comma-separated, +91… format) |
| telegram | `TELEGRAM_BOT_TOKEN` (from @BotFather), `TELEGRAM_CHAT_ID` |

Twilio trial accounts only text verified numbers, and India's DLT rules can delay or block
SMS. Keep Telegram ready as the demo fallback.

## Database

PostgreSQL 16 + PostGIS 3.4 has four tables: `zones`, `zone_readings`, `incidents` and
`alert_logs` ([app/models.py](backend/app/models.py)). Migrations are in
`backend\alembic\`:

```powershell
alembic upgrade head          # create/upgrade
alembic upgrade head --sql    # just print the SQL
```

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
| `backend/ml/` | Density maps, dataset prep, training, evaluation | §5, §8 |
| `backend/tools/` | Frame extraction, annotator, CVAT import, alert test, offline video analysis | Phase 1 |
| `config/zones.json` | Cameras, ROIs, areas/calibration, exit-graph nodes | |

## Ethics and privacy

- The system processes density maps, not identities: CSRNet outputs a heat map of how many
  people are where, and no face recognition or tracking is done.
- Only aggregate readings (counts, densities, risk levels) are stored. Frames live in Redis
  for seconds and are never written to the database.
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
