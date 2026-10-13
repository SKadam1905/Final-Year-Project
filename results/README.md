# Results (Phase 5)

Every table is a CSV plus a Markdown copy with the same name. Every figure is a 200 dpi PNG.
Scripts live in `scripts\` and run from the project root with `venv\Scripts\python.exe`.
Chapter numbers follow the plan's thesis structure (§9): Ch. 4 Implementation, Ch. 5 Dataset &
Training, Ch. 6 Results & Evaluation, Ch. 7 Conclusion & Future Work.

**Status (9 Oct 2026, 02:00):** ShanghaiTech is downloaded and verified, and the density maps are
built and checked. Our own CSRNet is training (`scripts\train_full.ps1`; follow it with
`scripts\check_training.ps1`). The active weights are the authors' public Part A model,
evaluated once on the 182 test images: MAE 71.8, RMSE 120.2. The UMN outputs wait for the
dataset.

Kept out of git (`.gitignore`): `*.pth`, `raw/`, the training logs and PID file, and
`kumbh_check/*.jpg` (frames of news footage with identifiable faces).

## Measured

| File | What it shows | Produced by | Thesis |
|---|---|---|---|
| `yolo_benchmark.csv/.md` | YOLOv8 n/s/m person detection (imgsz 640, FP16) next to CSRNet on the three demo clips in `data/india_dataset/raw_videos` (40 frames each): median/p95 ms per frame and people found per frame. yolov8s: 19.3 ms; CSRNet: 102.1 ms. YOLO finds 0 people per frame in the aerial clip (CSRNet 122), which is why `HYBRID_RULE=agree` is the default. | `yolo_benchmark.py` | Ch. 4 (hybrid counting), Ch. 6 (performance) |
| `yolo_benchmark_1280.csv/.md` | The same for yolov8s/m at imgsz 1280: 57.1 / 141.0 ms, still 0.2 / 0.1 people per aerial frame. | `yolo_benchmark.py --imgsz 1280` | Ch. 6 |
| `flow_speed_vs_accuracy.csv/.md/.png` | Optical-flow time per 5-frame burst at 1280, 640, 480, 320 and 240 px, and agreement with the full-resolution turbulence score (Pearson r, mean/max difference). Re-timed on 8 Oct 2026 with the backend stopped, including camera-motion and scene-cut checks: 320 px takes 50.4 ms (18.2× faster than full resolution, 4.3× faster than 640 px) at r = 0.998. This is the evidence for `FLOW_MAX_WIDTH=320`. | `flow_benchmark.py` | Ch. 4 (burst flow design), Ch. 6 (performance) |
| `flow_turbulence_by_regime.csv/.md` | Mean turbulence per motion regime of the synthetic clip at each width: does the metric rank still, orderly, counter-flow and chaotic motion as intended? | `flow_benchmark.py` | Ch. 4 (turbulence metric) |
| `flow_turbulence_per_burst.csv` | Every burst's score and time at every width (the raw data behind the two tables above). | `flow_benchmark.py` | Appendix |
| `stress_test.csv/.md/.png` | The live pipeline (`FRAME_SOURCE=local`) with 1, 2, 4 and 8 cameras for 2 minutes each: mean/p95 cycle latency against the 2 s deadline, late cycles, missed updates, GPU memory, CPU. Configurations: flow 640 px (before), flow 320 px (final), and flow 320 px with CSRNet at 960 px. Maximum cameras within the cycle: 2 at CSRNet 1280 px (4 cameras: p95 1.81 s, but 1 of 60 cycles late) and 4 at CSRNet 960 px. | `stress_test.py` (`--replot` redraws) | Ch. 6 (scalability), Ch. 7 (limits) |
| `kumbh_check.csv/.md`, `kumbh_check/*_samples.csv` | Counts with the active (public) weights on the two uploaded Kumbh news clips: count range and variation per clip, scene cuts, moving-camera samples, and whole-frame vs density-weighted turbulence. Includes a visual check: the counts follow the scene but are far too low on dense and aerial shots (a domain gap, not a code fault). | `kumbh_check.py` | Ch. 6 (qualitative), Ch. 7 (limits) |
| `kumbh_check/*_grid.jpg` (not in git) | Six frames per clip with the predicted count and where CSRNet placed people. | `kumbh_check.py` | Ch. 6 (figures; blur faces before printing) |
| `INDIA_FINETUNE.md` | Step-by-step guide to collecting, annotating (CVAT), splitting by video, fine-tuning and evaluating on Indian footage. | (written) | Ch. 5 (method) |
| `CALIBRATION_GUIDE.md` | Measuring a rectangle at Ramkund with Google Maps' measure tool (or a tape), and calibrating a zone with it. | (written) | Ch. 4 (calibration method) |
| `density_map_check.csv/.md` | For ShanghaiTech Part A (adaptive kernels) and Part B (sigma 15), per split: images, heads, and the worst gap between a density map's sum and its head count (2.7e-07%; the limit is 1%). | `check_density_maps.py` | Ch. 5 (ground truth) |
| `density_map_examples.png` | Three Part A images and one Part B image with their ground-truth density maps overlaid. | `check_density_maps.py` | Ch. 5 |
| `training_vram.csv/.md` | Peak GPU memory and time per step for each batch × crop size on the 4 GB card. Largest that fit: 8 × 512 px and 12 × 384 px; training uses 8 × 384 px (1.36 s per step). | `vram_probe.py` | Ch. 5 (training setup) |
| `public_pretrained/csrnet_eval.csv/.md`, `csrnet_summary.csv/.md`, `csrnet_scatter.png`, `csrnet_samples.png` | The authors' public Part A weights, evaluated once on the 182 test images: MAE 71.8, RMSE 120.2 (paper 68.2 / 115.0). Per-image counts, predicted vs true with y = x, and best/median/worst images. | `report_csrnet.py --out results\public_pretrained --record-provenance` | Ch. 6 (baseline) |

## Waiting for the dataset or labels

These scripts are ready and tested on synthetic data. The datasets are described in
[../data/README.md](../data/README.md).

| File | What it will show | How to produce it | Thesis |
|---|---|---|---|
| `train_log.csv` (being written), `training_curve.png` | One row per epoch (train/val loss, val MAE/RMSE, learning rate, seconds), and the curves against the paper's 68.2. | `scripts\train_full.ps1` (running), then `plot_training.py` | Ch. 5 |
| `csrnet_eval.csv/.md`, `csrnet_summary.csv/.md`, `csrnet_scatter.png`, `csrnet_samples.png` | Our trained model on the 182 Part A test images: MAE/RMSE next to Li et al. (68.2 / 115.0) and the public weights (71.8 / 120.2), predicted vs true with y = x, and best/median/worst images. | `report_csrnet.py --weights backend\ml\weights\parta_best.pth --record-provenance` (once, after training) | Ch. 6 |
| `turbulence_roc.csv/.md/.png`, `turbulence_weighting_auc.csv/.md` | ROC/AUC of the turbulence score as a panic detector on UMN, per scene and overall, with the Youden threshold against the 0.3/0.6 bands. With `--compare-density`, the AUC of whole-frame vs density-weighted turbulence. | `turbulence_roc.py [--compare-density]` (needs `data\umn\`) | Ch. 6 |
| `india_before/`, `india_after/`, `india_finetune.csv/.md` | Counting error on held-out Indian test videos before and after fine-tuning. | `india_finetune.ps1 -TestVideos … -ValVideos …` | Ch. 6 |
| `zone_calibration.md`, `zone_calibration_readings.csv` | Per zone, the estimated vs calibrated ground area and the number of people at which each level starts; with `--run`, people counted, persons/m² and risk levels on the zone's clip, estimated vs calibrated. | `python -m tools.calibration_report --run` from `backend\`, after `tools.calibrate_zone` with a measured rectangle (no zone is calibrated yet) | Ch. 4 (calibration), Ch. 6 |
