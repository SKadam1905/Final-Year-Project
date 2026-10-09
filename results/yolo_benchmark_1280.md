# YOLOv8 person detection vs CSRNet: time per frame on the demo clips

| model | frames | median_ms | p95_ms | mean_count_kumbh_aerial_news9 | mean_count_prayagraj_ghat_pti | mean_count_prayagraj_sangam_pti | est_cycle_s_3_zones | fits_2s |
|---|---|---|---|---|---|---|---|---|
| CSRNet (csrnet_partA_public.pth) | 120 | 101.8 | 104.1 | 122.0 | 91.3 | 96.3 |  |  |
| yolov8s (imgsz 1280) | 120 | 57.1 | 59.4 | 0.2 | 21.6 | 16.3 | 0.48 | True |
| yolov8m (imgsz 1280) | 120 | 141.0 | 142.9 | 0.1 | 16.1 | 10.5 | 0.73 | True |

- Device: NVIDIA GeForce RTX 2050. 40 evenly spaced frames from each of 3 clips in data/india_dataset/raw_videos (kumbh_aerial_news9, prayagraj_ghat_pti, prayagraj_sangam_pti), 640x360. One frame at a time, after 5 warm-up calls.
- YOLO: COCO-pretrained, person class only, conf 0.25, half precision on the GPU. CSRNet: INFER_MAX_WIDTH=1280, AMP.
- est_cycle_s = zones x (YOLO median + CSRNet median): GPU time of one live cycle, before optical flow (which runs on the CPU, in parallel across zones). The measured cycle time is /health last_cycle_s.
- mean_count_* = mean number of people found per frame in that clip (no ground truth for these clips).
- Command: venv\Scripts\python.exe scripts\yolo_benchmark.py
