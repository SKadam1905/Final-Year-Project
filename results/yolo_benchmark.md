# YOLOv8 person detection vs CSRNet: time per frame on the demo clips

| model | frames | median_ms | p95_ms | mean_count_kumbh_aerial_news9 | mean_count_prayagraj_ghat_pti | mean_count_prayagraj_sangam_pti | est_cycle_s_3_zones | fits_2s |
|---|---|---|---|---|---|---|---|---|
| CSRNet (csrnet_partA_public.pth) | 120 | 102.1 | 106.0 | 122.0 | 91.3 | 96.3 |  |  |
| yolov8n (imgsz 640) | 120 | 13.0 | 28.5 | 0.0 | 10.2 | 5.2 | 0.35 | True |
| yolov8s (imgsz 640) | 120 | 19.3 | 23.1 | 0.0 | 14.3 | 7.5 | 0.36 | True |
| yolov8m (imgsz 640) | 120 | 43.8 | 45.0 | 0.0 | 11.4 | 6.6 | 0.44 | True |

- Device: NVIDIA GeForce RTX 2050. 40 evenly spaced frames from each of 3 clips in data/india_dataset/raw_videos (kumbh_aerial_news9, prayagraj_ghat_pti, prayagraj_sangam_pti), 640x360. One frame at a time, after 5 warm-up calls.
- YOLO: COCO-pretrained, person class only, conf 0.25, half precision on the GPU. CSRNet: INFER_MAX_WIDTH=1280, AMP.
- est_cycle_s = zones x (YOLO median + CSRNet median): GPU time of one live cycle, before optical flow (which runs on the CPU, in parallel across zones). The measured cycle time is /health last_cycle_s.
- mean_count_* = mean number of people found per frame in that clip (no ground truth for these clips).
- Command: venv\Scripts\python.exe scripts\yolo_benchmark.py
