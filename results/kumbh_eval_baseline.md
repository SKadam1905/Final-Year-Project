# Counting accuracy on Kumbh footage: baseline (step 2)

14 frame(s) with complete human ground truth, shot types: aerial, close, medium, wide. Each model is scored inside the frame's counting region; see the notes before reading the numbers.

## Error per frame

| frame | shot_type | region (x0,y0,x1,y1) | true (annotators' mean) | CSRNet | CSRNet err % | YOLO | YOLO err % | hybrid | hybrid err % | hybrid uses |
|---|---|---|---|---|---|---|---|---|---|---|
| frame_01.png | aerial | 100,0,205,360 | 51 | 3.1 | 93.9 | 0.0 | 100.0 | 3.1 | 93.9 | CSRNet |
| frame_02.png | aerial | 0,0,250,85 | 57 | 5.9 | 89.6 | 0.0 | 100.0 | 5.9 | 89.6 | CSRNet |
| frame_04.png | aerial | 300,120,410,360 | 46.5 | 7.5 | 83.8 | 0.0 | 100.0 | 7.5 | 83.8 | CSRNet |
| frame_05.png | wide | 0,230,640,360 | 48 | 40.3 | 16.0 | 11.0 | 77.1 | 40.3 | 16.0 | CSRNet |
| frame_06.png | wide | 0,200,640,360 | 45.5 | 50.7 | 11.4 | 4.0 | 91.2 | 50.7 | 11.4 | CSRNet |
| frame_07.png | wide | 0,230,640,360 | 45.5 | 53.6 | 17.9 | 6.0 | 86.8 | 53.6 | 17.9 | CSRNet |
| frame_08.png | medium | 0,185,640,360 | 35 | 48.4 | 38.2 | 13.0 | 62.9 | 48.4 | 38.2 | CSRNet |
| frame_09.png | medium | 280,140,640,360 | 16.5 | 9.1 | 44.7 | 12.0 | 27.3 | 12.0 | 27.3 | YOLO |
| frame_10.png | medium | 0,0,640,360 | 17.5 | 18.4 | 4.9 | 17.0 | 2.9 | 17.0 | 2.9 | YOLO |
| frame_11.png | medium | 0,0,640,360 | 14 | 9.7 | 30.8 | 13.0 | 7.1 | 13.0 | 7.1 | YOLO |
| frame_12.png | close | 0,0,640,360 | 22 | 50.7 | 130.6 | 12.0 | 45.5 | 12.0 | 45.5 | YOLO |
| frame_13.png | close | 0,0,640,360 | 13 | 35.9 | 175.9 | 13.0 | 0.0 | 13.0 | 0.0 | YOLO |
| frame_14.png | close | 0,0,640,360 | 8 | 11.0 | 37.8 | 10.0 | 25.0 | 10.0 | 25.0 | YOLO |
| frame_15.png | close | 0,0,640,360 | 5 | 6.8 | 36.1 | 3.0 | 40.0 | 3.0 | 40.0 | YOLO |

## Error per shot type

| method | shot_type | frames | true_heads | MAE | relative error % | ms per frame |
|---|---|---|---|---|---|---|
| CSRNet | aerial | 3 | 154.5 | 46.0 | 89.3 | 116.2 |
| CSRNet | close | 4 | 48 | 14.1 | 117.6 | 116.2 |
| CSRNet | medium | 4 | 83 | 6.5 | 31.2 | 116.2 |
| CSRNet | wide | 3 | 139 | 7.0 | 15.1 | 116.2 |
| CSRNet | all | 14 | 424.5 | 17.2 | 56.8 | 116.2 |
| YOLO | aerial | 3 | 154.5 | 51.5 | 100.0 | 37.8 |
| YOLO | close | 4 | 48 | 3.5 | 29.2 | 37.8 |
| YOLO | medium | 4 | 83 | 7.0 | 33.7 | 37.8 |
| YOLO | wide | 3 | 139 | 39.3 | 84.9 | 37.8 |
| YOLO | all | 14 | 424.5 | 22.5 | 74.1 | 37.8 |
| hybrid | aerial | 3 | 154.5 | 46.0 | 89.3 | 153.9 |
| hybrid | close | 4 | 48 | 3.5 | 29.2 | 153.9 |
| hybrid | medium | 4 | 83 | 4.8 | 23.3 | 153.9 |
| hybrid | wide | 3 | 139 | 7.0 | 15.1 | 153.9 |
| hybrid | all | 14 | 424.5 | 13.7 | 45.3 | 153.9 |

## Inter-annotator agreement

| frame | shot_type | rohit | shubham | mean | annotators | difference % |
|---|---|---|---|---|---|---|
| frame_01.png | aerial | 34 | 68 | 51.0 | 2 | 66.7 |
| frame_02.png | aerial | 46 | 68 | 57.0 | 2 | 38.6 |
| frame_04.png | aerial | 50 | 43 | 46.5 | 2 | 15.1 |
| frame_05.png | wide | 45 | 51 | 48.0 | 2 | 12.5 |
| frame_06.png | wide | 38 | 53 | 45.5 | 2 | 33.0 |
| frame_07.png | wide | 41 | 50 | 45.5 | 2 | 19.8 |
| frame_08.png | medium | 34 | 36 | 35.0 | 2 | 5.7 |
| frame_09.png | medium | 15 | 18 | 16.5 | 2 | 18.2 |
| frame_10.png | medium | 17 | 18 | 17.5 | 2 | 5.7 |
| frame_11.png | medium | 14 | 14 | 14.0 | 2 | 0.0 |
| frame_12.png | close | 27 | 17 | 22.0 | 2 | 45.5 |
| frame_13.png | close | 13 | 13 | 13.0 | 2 | 0.0 |
| frame_14.png | close | 7 | 9 | 8.0 | 2 | 25.0 |
| frame_15.png | close | 6 | 4 | 5.0 | 2 | 40.0 |

Each person's total for the frame's region, the mean used as the true count, and the difference: (highest - lowest) / mean, which for two people is their % difference. Frames counted by two or more people: 14 of 14; mean difference 23.3%.

## Notes

- **Ground truth**: the mean of the annotators' totals; each person counted the heads per cell of a 3x3 grid over the frame's counting region (`tools.count_eval --annotator <name>`, `data/eval/annotations/kumbh_eval/counts_<name>.csv`), without seeing anyone else's counts; no model suggested any count.
- **Count error only**: the ground truth has no head positions, so point-level localisation is not evaluated. err % = |predicted - true| / true per frame; relative error % = sum |error| / sum true heads.
- **CSRNet**: Public pretrained CSRNet (ShanghaiTech Part A, github.com/leeyeehoo/CSRNet-pytorch, Google Drive id 1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH); the authors report MAE 66.4; test MAE 71.8 / RMSE 120.2 measured here. At INFER_MAX_WIDTH=1280 (these 640x360 frames run at 640x360: frames are only ever downscaled); density summed over the map cells whose centres lie in the region.
- **YOLO**: yolov8s, person class, imgsz 640, conf 0.25 (ultralytics COCO-pretrained (person class only)); a person counts in the region by the estimated head position (box centre, 0.2 x box width below its top), since the ground truth counts heads.
- **Hybrid**: `choose_count` with HYBRID_RULE=agree, HYBRID_THRESHOLD=60, decided as live on the whole-frame counts (shown below), then scored on the chosen model's count in the region.
- **Time**: mean model time per frame after warm-up on cuda (NVIDIA GeForce RTX 2050), one frame at a time; the hybrid runs both models. 3 cameras in the 2 s cycle need 3 x 154 ms = 0.46 s of the hybrid's model time (optical flow and the rest of the cycle not included).
- **Left out**: frame_03.png: left out, marked unreadable (heads too small or dense to separate).
- frame_01.png: counts used; the point file frame_01.json is ignored.

## Observations (computed from the table)

- CSRNet predicted fewer heads than counted on 6 of 14 frames, YOLO on 12, the hybrid on 9.
- YOLO found nobody inside the region on 3 of 14 frames.
- The hybrid used CSRNet's count on 7 of 14 frames and YOLO's on 7.

Whole-frame counts behind each hybrid decision:

| frame | whole_frame_csrnet | whole_frame_yolo | hybrid_uses |
|---|---|---|---|
| frame_01.png | 98.1 | 0 | CSRNet |
| frame_02.png | 220.9 | 0 | CSRNet |
| frame_04.png | 65.6 | 0 | CSRNet |
| frame_05.png | 164.6 | 12 | CSRNet |
| frame_06.png | 78.7 | 5 | CSRNet |
| frame_07.png | 133.4 | 6 | CSRNet |
| frame_08.png | 126.6 | 13 | CSRNet |
| frame_09.png | 11.6 | 12 | YOLO |
| frame_10.png | 18.4 | 17 | YOLO |
| frame_11.png | 9.7 | 13 | YOLO |
| frame_12.png | 50.7 | 12 | YOLO |
| frame_13.png | 35.9 | 13 | YOLO |
| frame_14.png | 11.0 | 10 | YOLO |
| frame_15.png | 6.8 | 3 | YOLO |

