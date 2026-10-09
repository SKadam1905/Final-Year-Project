# Counts on real Kumbh footage (active weights)

| clip | samples | analysed | scene_cut | moving_camera | count_min | count_median | count_max | distinct_counts | count_cv_pct | turbulence_frame_mean | turbulence_density_mean | turbulence_mean_abs_diff | turbulence_pearson_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Maha Kumbh AI-enabled cameras drones monitor crowd and traffic in Prayagraj [get.gt] | 57 | 55 | 2 | 8 | 5 | 11 | 183 | 32 | 115.2 | 0.222 | 0.172 | 0.054 | 0.893 |
| test_video | 67 | 66 | 1 | 26 | 3 | 143 | 323 | 55 | 57.5 | 0.118 | 0.092 | 0.034 | 0.924 |

- Weights: Public pretrained CSRNet (ShanghaiTech Part A, github.com/leeyeehoo/CSRNet-pytorch, Google Drive id 1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH); the authors report MAE 66.4; not yet evaluated on this machine.
- No ground truth for these clips: the counts are for checking by eye with the grids in results/kumbh_check/; they are not an accuracy measurement.
- Sampling as in video analysis: one 5-frame burst every 2 s of video; bursts spanning a scene cut skipped. count_cv_pct = standard deviation / mean of the counts.
- turbulence_frame = every moving pixel counts (the default); turbulence_density = flow weighted by the CSRNet density map (TURBULENCE_WEIGHTING=density). Camera motion is removed first in both.

## What the grids show (visual check, not a measurement)

Weights: the public CSRNet ShanghaiTech Part A model (Li et al.), imported with
`backend/tools/import_public_weights.py`. It is not evaluated on the Part A test set here: the
dataset is not on this machine.

- **The counts follow the scene but are far too low on dense crowds.** In the Prayagraj clip, a
  street packed edge to edge at 0:22 gets 135 and the bathers at 1:30 get 76. In the aerial
  `test_video`, drone shots with thousands of people per frame get 69-306. The heatmaps mark only
  scattered small heads.
- **Why.** Part A is oblique photos (about 1024x768) with small, sharp heads. These clips are
  640x360 news video: aerial views from straight above (`test_video`, all frames), people close to
  the camera (Prayagraj 0:00, 0:44, 1:08), haze and motion blur.
- **It is not the input resolution.** Enlarging the 640x360 frames 1.5x, 2x and 3x before CSRNet
  lowers the count instead of raising it (Prayagraj 0:22: 135 -> 88 -> 86 -> 74).
- **It is not our code.** fp16 and fp32 inference, and the authors' own preprocessing (PIL RGB,
  ToTensor, ImageNet normalisation), give identical counts (114.9 / 135.0 / 75.9 on the three
  frames tested).
- **Non-crowd shots still get a count:** the interview close-up (1 person -> 8), and the PTI and
  News9 end cards (5 and 3).
- **Which clips to distrust:** every aerial shot in `test_video`, and in Prayagraj the close-up
  street shots, the interview and the end card.
- **What fixes it:** fine-tuning on annotated Indian footage (results/INDIA_FINETUNE.md), with
  aerial and oblique views in separate test videos. Until then, treat these counts as relative
  (whether a scene is getting denser), not absolute.

## Density-weighted turbulence (step 4)

The two variants agree closely: Pearson r 0.89 and 0.92, mean absolute difference 0.054 and 0.034.
Density weighting gives lower averages (Prayagraj 0.222 -> 0.172; `test_video` 0.118 -> 0.092),
as it should when empty areas such as water and sky drop out. Whether that makes it *better*
needs labelled panic/normal footage. The UMN dataset is not on this machine, so no AUC could be
computed.

With these weights the density maps miss most of the people, so weighting by them would also
ignore regions that do hold people. The default therefore stays `TURBULENCE_WEIGHTING=frame`. To
change it, run `scripts/turbulence_roc.py` once UMN and real trained weights are available, compare
the AUC of both variants, and switch to `density` only if it is at least as good.
