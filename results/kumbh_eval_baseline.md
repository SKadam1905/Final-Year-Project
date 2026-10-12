# Counting accuracy on Kumbh footage: baseline (step 2)

3 frame(s) with complete human ground truth, shot types: aerial. Each model is scored inside the frame's counting region; see the notes before reading the numbers.

> **Too little ground truth to compare methods in general.** Only 3 frame(s), all aerial, are counted completely; the other shot types have no ground truth yet, so this table says nothing about them, and nothing about dense crowds (the aerial regions are the sparse sand and water areas). Count more frames with `tools.count_eval` and run this again.

## Error per frame

| frame | shot_type | region (x0,y0,x1,y1) | true | CSRNet | CSRNet err % | YOLO | YOLO err % | hybrid | hybrid err % | hybrid uses |
|---|---|---|---|---|---|---|---|---|---|---|
| frame_01.png | aerial | 100,0,205,360 | 68 | 3.1 | 95.4 | 0.0 | 100.0 | 3.1 | 95.4 | CSRNet |
| frame_02.png | aerial | 0,0,250,85 | 68 | 5.9 | 91.3 | 0.0 | 100.0 | 5.9 | 91.3 | CSRNet |
| frame_04.png | aerial | 300,120,410,360 | 43 | 7.5 | 82.5 | 0.0 | 100.0 | 7.5 | 82.5 | CSRNet |

## Error per shot type

| method | shot_type | frames | true_heads | MAE | relative error % | ms per frame |
|---|---|---|---|---|---|---|
| CSRNet | aerial | 3 | 179 | 54.1 | 90.7 | 106.8 |
| YOLO | aerial | 3 | 179 | 59.7 | 100.0 | 34.5 |
| hybrid | aerial | 3 | 179 | 54.1 | 90.7 | 141.3 |

## Notes

- **Ground truth**: heads counted by a person per cell of a 3x3 grid over each frame's counting region (`tools.count_eval`, `data/eval/annotations/kumbh_eval/counts.csv`); no model suggested any count.
- **Count error only**: the ground truth has no head positions, so point-level localisation is not evaluated. err % = |predicted - true| / true per frame; relative error % = sum |error| / sum true heads.
- **CSRNet**: Public pretrained CSRNet (ShanghaiTech Part A, github.com/leeyeehoo/CSRNet-pytorch, Google Drive id 1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH); the authors report MAE 66.4; test MAE 71.8 / RMSE 120.2 measured here. At INFER_MAX_WIDTH=1280 (these 640x360 frames run at 640x360: frames are only ever downscaled); density summed over the map cells whose centres lie in the region.
- **YOLO**: yolov8s, person class, imgsz 640, conf 0.25 (ultralytics COCO-pretrained (person class only)); a person counts in the region by the estimated head position (box centre, 0.2 x box width below its top), since the ground truth counts heads.
- **Hybrid**: `choose_count` with HYBRID_RULE=agree, HYBRID_THRESHOLD=60, decided as live on the whole-frame counts (shown below), then scored on the chosen model's count in the region.
- **Time**: mean model time per frame after warm-up on cuda (NVIDIA GeForce RTX 2050), one frame at a time; the hybrid runs both models. 3 cameras in the 2 s cycle need 3 x 141 ms = 0.42 s of the hybrid's model time (optical flow and the rest of the cycle not included).
- **Left out**: frame_03.png: left out, marked unreadable (heads too small or dense to separate); frame_05.png: left out, not annotated; frame_06.png: left out, not annotated; frame_07.png: left out, not annotated; frame_08.png: left out, not annotated; frame_09.png: left out, not annotated; frame_10.png: left out, not annotated; frame_11.png: left out, not annotated; frame_12.png: left out, not annotated; frame_13.png: left out, not annotated; frame_14.png: left out, not annotated; frame_15.png: left out, not annotated.
- frame_01.png: counts.csv used; the point file frame_01.json is ignored.

## Observations (computed from the table)

- CSRNet predicted fewer heads than counted on 3 of 3 frames, YOLO on 3, the hybrid on 3.
- YOLO found nobody inside the region on 3 of 3 frames.
- The hybrid used CSRNet's count on 3 of 3 frames and YOLO's on 0.

Whole-frame counts behind each hybrid decision:

| frame | whole_frame_csrnet | whole_frame_yolo | hybrid_uses |
|---|---|---|---|
| frame_01.png | 98.1 | 0 | CSRNet |
| frame_02.png | 220.9 | 0 | CSRNet |
| frame_04.png | 65.6 | 0 | CSRNet |

