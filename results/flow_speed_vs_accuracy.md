# Optical flow: speed against accuracy by flow width

| flow_width_px | mean_ms_per_burst | p95_ms_per_burst | speedup_vs_full | speedup_vs_640 | pearson_r_vs_full | mean_abs_diff_vs_full | max_abs_diff_vs_full | meets_r_0.95 |
|---|---|---|---|---|---|---|---|---|
| 1280 (full) | 916.8 | 994.5 | 1.0 | 0.24 | 1.0 | 0.0 | 0.0 | yes |
| 640 | 216.7 | 233.6 | 4.23 | 1.0 | 0.9998 | 0.0089 | 0.0162 | yes |
| 480 | 120.1 | 133.2 | 7.64 | 1.81 | 0.9996 | 0.012 | 0.029 | yes |
| 320 | 50.4 | 56.1 | 18.19 | 4.3 | 0.998 | 0.0296 | 0.0725 | yes |
| 240 | 32.3 | 36.6 | 28.41 | 6.72 | 0.998 | 0.0349 | 0.0852 | yes |

- Clip: synthetic_crowd_1280.mp4 (1280x720, 133 bursts of 5 consecutive frames, one every 12 frames). Synthetic crowd with eight motion regimes; see scripts/make_test_clip.py.
- Time per burst = ml.optical_flow.burst_turbulence (downscale, cut check, 4 Farneback pairs, turbulence) on one CPU thread calling OpenCV (12 OpenCV threads).
- Agreement is with the full-resolution score over the same bursts. Widths meeting r >= 0.95: 640, 480, 320, 240. Rule: keep 320 px if it meets the bar, otherwise use the smallest width that does. Chosen FLOW_MAX_WIDTH: 320 px.
- The clip's regimes span turbulence 0-1, which helps the correlation; the mean absolute difference is the better measure of error. Re-check on real footage once it is available.
