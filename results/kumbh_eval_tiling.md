# Counting accuracy on Kumbh footage: pretrained vs tiled / upscaled (step 3)

14 frames with complete human ground truth (mean of the annotators' counts inside each frame's counting region), shot types: aerial, close, medium, wide. No model was fine-tuned and the live pipeline is unchanged: the tiled / upscaled variants exist only in `app/tiled_inference.py` and this script.

## Pretrained vs tiled / upscaled, per shot type

MAE in heads (relative error = sum |error| / sum true heads):

| shot type | frames | true heads | CSRNet 1x | CSRNet 2x | CSRNet 3x | YOLO | YOLO sliced | hybrid (pretrained) | hybrid 1x + sliced | hybrid 2x + sliced | hybrid 3x + sliced |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aerial | 3 | 154.5 | 46.0 (89.3%) | 46.2 (89.6%) | 45.6 (88.6%) | 51.5 (100.0%) | 51.5 (100.0%) | 46.0 (89.3%) | 46.0 (89.3%) | 47.8 (92.9%) | 47.7 (92.5%) |
| close | 4 | 48 | 14.1 (117.6%) | 22.9 (190.8%) | 28.2 (235.4%) | 3.5 (29.2%) | 3.0 (25.0%) | 3.5 (29.2%) | 3.0 (25.0%) | 11.1 (92.3%) | 11.8 (98.2%) |
| medium | 4 | 83 | 6.5 (31.2%) | 3.2 (15.6%) | 2.6 (12.4%) | 7.0 (33.7%) | 4.8 (22.9%) | 4.8 (23.3%) | 4.3 (20.9%) | 1.6 (7.7%) | 1.9 (9.0%) |
| wide | 3 | 139 | 7.0 (15.1%) | 9.6 (20.8%) | 14.7 (31.7%) | 39.3 (84.9%) | 24.0 (51.8%) | 7.0 (15.1%) | 7.0 (15.1%) | 15.4 (33.2%) | 22.6 (48.9%) |
| all | 14 | 424.5 | 17.2 (56.8%) | 19.4 (64.1%) | 21.7 (71.7%) | 22.5 (74.1%) | 18.4 (60.7%) | 13.7 (45.3%) | 13.5 (44.4%) | 17.2 (56.6%) | 19.0 (62.5%) |
| **ms per frame** (median pass) |  |  | 136 | 467 | 1138 | 54 | 540 | 190 | 657 | 998 | 1654 |
| **4 cameras in 2 s** |  |  | yes | borderline | no | yes | no | yes | no | no | no |

Bias (sum of predicted - true over sum of true heads; negative = undercount):

| shot type | CSRNet 1x | CSRNet 2x | CSRNet 3x | YOLO | YOLO sliced | hybrid (pretrained) | hybrid 1x + sliced | hybrid 2x + sliced | hybrid 3x + sliced |
|---|---|---|---|---|---|---|---|---|---|
| aerial | -89.3% | -89.6% | -88.6% | -100.0% | -100.0% | -89.3% | -89.3% | -92.9% | -92.5% |
| close | +117.6% | +190.8% | +235.4% | -20.8% | -20.8% | -20.8% | -20.8% | +88.1% | +94.0% |
| medium | +3.0% | -2.9% | +1.0% | -33.7% | -14.5% | +8.9% | +19.7% | +6.5% | -0.6% |
| wide | +4.1% | -20.8% | -31.7% | -84.9% | -51.8% | +4.1% | +4.1% | -33.2% | -48.9% |
| all | -17.3% | -18.4% | -15.8% | -73.1% | -58.5% | -31.8% | -29.7% | -33.5% | -39.2% |

## Time per frame

| variant | ms per frame (median pass) | range over passes | 4 cameras (s), slowest pass | within the 2 s cycle | kind |
|---|---|---|---|---|---|
| CSRNet 1x | 135.6 | 117-194 | 0.78 | yes | pretrained, as deployed |
| CSRNet 2x | 467.4 | 459-544 | 2.18 | borderline | tiled / upscaled |
| CSRNet 3x | 1138.3 | 1114-1224 | 4.9 | no | tiled / upscaled |
| YOLO | 54.1 | 34-59 | 0.24 | yes | pretrained, as deployed |
| YOLO sliced | 539.6 | 217-727 | 2.91 | no | tiled / upscaled |
| hybrid (pretrained) | 189.7 | 176-228 | 0.91 | yes | pretrained, as deployed |
| hybrid 1x + sliced | 656.8 | 410-862 | 3.45 | no | tiled / upscaled |
| hybrid 2x + sliced | 998.5 | 761-1194 | 4.78 | no | tiled / upscaled |
| hybrid 3x + sliced | 1653.7 | 1355-1951 | 7.8 | no | tiled / upscaled |

Model time per 640x360 frame on cuda (NVIDIA GeForce RTX 2050), each model variant timed on its own over all frames (warm-up, then the median of 3 runs per frame), in 3 separate passes; a hybrid's time is the sum of its two models' in the same pass. yes = even the slowest pass fits; borderline = the median pass fits but the slowest does not. 4 cameras in the 2 s cycle leave 500 ms per frame for everything; optical flow and the rest of the cycle come on top of the model time shown. Times on this laptop GPU vary between passes (power and clock states): hence the range.

## Error per frame

| frame | shot_type | true (annotators' mean) | CSRNet 1x | CSRNet 2x | CSRNet 3x | YOLO | YOLO sliced | hybrid (pretrained) | hybrid 1x + sliced | hybrid 2x + sliced | hybrid 3x + sliced |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frame_01.png | aerial | 51 | 3.1 | 3.6 | 4.9 | 0.0 | 0.0 | 3.1 | 3.1 | 3.6 | 4.9 |
| frame_02.png | aerial | 57 | 5.9 | 7.3 | 6.7 | 0.0 | 0.0 | 5.9 | 5.9 | 7.3 | 6.7 |
| frame_04.png | aerial | 46.5 | 7.5 | 5.0 | 6.1 | 0.0 | 0.0 | 7.5 | 7.5 | 0.0 | 0.0 |
| frame_05.png | wide | 48 | 40.3 | 34.5 | 29.1 | 11.0 | 25.0 | 40.3 | 40.3 | 34.5 | 29.1 |
| frame_06.png | wide | 45.5 | 50.7 | 35.2 | 30.1 | 4.0 | 18.0 | 50.7 | 50.7 | 18.0 | 18.0 |
| frame_07.png | wide | 45.5 | 53.6 | 40.3 | 35.8 | 6.0 | 24.0 | 53.6 | 53.6 | 40.3 | 24.0 |
| frame_08.png | medium | 35 | 48.4 | 37.4 | 31.5 | 13.0 | 20.0 | 48.4 | 48.4 | 37.4 | 31.5 |
| frame_09.png | medium | 16.5 | 9.1 | 11.9 | 15.2 | 12.0 | 16.0 | 12.0 | 16.0 | 16.0 | 16.0 |
| frame_10.png | medium | 17.5 | 18.4 | 20.4 | 22.5 | 17.0 | 18.0 | 17.0 | 18.0 | 18.0 | 18.0 |
| frame_11.png | medium | 14 | 9.7 | 10.9 | 14.5 | 13.0 | 17.0 | 13.0 | 17.0 | 17.0 | 17.0 |
| frame_12.png | close | 22 | 50.7 | 64.3 | 67.1 | 12.0 | 12.0 | 12.0 | 12.0 | 64.3 | 67.1 |
| frame_13.png | close | 13 | 35.9 | 49.9 | 59.9 | 13.0 | 12.0 | 13.0 | 12.0 | 12.0 | 12.0 |
| frame_14.png | close | 8 | 11.0 | 13.0 | 13.8 | 10.0 | 9.0 | 10.0 | 9.0 | 9.0 | 9.0 |
| frame_15.png | close | 5 | 6.8 | 12.4 | 20.2 | 3.0 | 5.0 | 3.0 | 5.0 | 5.0 | 5.0 |

## Ground truth: inter-annotator agreement

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

Mean difference between annotators: 23.3% over 14 frames. A model error smaller than this cannot be told apart from counting noise.

## Notes

- **CSRNet**: Public pretrained CSRNet (ShanghaiTech Part A, github.com/leeyeehoo/CSRNet-pytorch, Google Drive id 1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH); the authors report MAE 66.4; test MAE 71.8 / RMSE 120.2 measured here. Fed at 1x, 2x and 3x the frame size (bicubic upscaling; 640x360 -> 1280x720 -> 1920x1080); density summed over the map cells whose centres lie in the region, mapped back to original pixels.
- **YOLO**: yolov8s, person class, imgsz 640, conf 0.25. Sliced: 6 tiles of 320 px with 20% overlap per 640x360 frame, each upscaled to 640 px (bicubic) and detected in one batch, plus the whole frame; boxes merged across tiles when the intersection covers >= 50% of the smaller box. People count in the region by their estimated head position.
- **Hybrid**: `choose_count` with HYBRID_RULE=agree, HYBRID_THRESHOLD=60 on the variant's whole-frame counts, scored on the chosen model's count in the region.
- **Count error only**: the ground truth has no head positions, so localisation is not evaluated.
- **Left out**: frame_03.png: left out, marked unreadable (heads too small or dense to separate).

Which model each hybrid used:

| frame | hybrid (pretrained) | hybrid 1x + sliced | hybrid 2x + sliced | hybrid 3x + sliced |
|---|---|---|---|---|
| frame_01.png | CSRNet 1x | CSRNet 1x | CSRNet 2x | CSRNet 3x |
| frame_02.png | CSRNet 1x | CSRNet 1x | CSRNet 2x | CSRNet 3x |
| frame_04.png | CSRNet 1x | CSRNet 1x | YOLO sliced | YOLO sliced |
| frame_05.png | CSRNet 1x | CSRNet 1x | CSRNet 2x | CSRNet 3x |
| frame_06.png | CSRNet 1x | CSRNet 1x | YOLO sliced | YOLO sliced |
| frame_07.png | CSRNet 1x | CSRNet 1x | CSRNet 2x | YOLO sliced |
| frame_08.png | CSRNet 1x | CSRNet 1x | CSRNet 2x | CSRNet 3x |
| frame_09.png | YOLO | YOLO sliced | YOLO sliced | YOLO sliced |
| frame_10.png | YOLO | YOLO sliced | YOLO sliced | YOLO sliced |
| frame_11.png | YOLO | YOLO sliced | YOLO sliced | YOLO sliced |
| frame_12.png | YOLO | YOLO sliced | CSRNet 2x | CSRNet 3x |
| frame_13.png | YOLO | YOLO sliced | YOLO sliced | YOLO sliced |
| frame_14.png | YOLO | YOLO sliced | YOLO sliced | YOLO sliced |
| frame_15.png | YOLO | YOLO sliced | YOLO sliced | YOLO sliced |

