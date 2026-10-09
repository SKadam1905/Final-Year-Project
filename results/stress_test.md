# Live-pipeline stress test

| config | flow_width_px | csrnet_width_px | cameras | cycles | mean_latency_s | p95_latency_s | max_latency_s | late_cycles_pct | all_zones_updated_pct | missed_zone_updates | max_update_gap_s | gpu_peak_allocated_mb | gpu_peak_reserved_mb | cpu_pct_of_one_core | cpu_pct_of_machine | meets_2s_cycle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before: flow 640 | 640 | 1280 | 1 | 60 | 0.769 | 0.923 | 0.938 | 0.0 | 100.0 | 0 | 2.25 | 590 | 618 | 50 | 4.2 | yes |
| before: flow 640 | 640 | 1280 | 2 | 60 | 1.117 | 1.344 | 1.593 | 0.0 | 98.3 | 2 | 4.16 | 653 | 692 | 111 | 9.2 | yes |
| before: flow 640 | 640 | 1280 | 4 | 59 | 1.935 | 2.141 | 2.406 | 25.4 | 100.0 | 0 | 2.55 | 589 | 752 | 254 | 21.1 | no |
| before: flow 640 | 640 | 1280 | 8 | 34 | 3.549 | 4.292 | 4.656 | 100.0 | 100.0 | 0 | 4.66 | 653 | 692 | 421 | 35.1 | no |
| after: flow 320 | 320 | 1280 | 1 | 60 | 0.604 | 0.767 | 0.797 | 0.0 | 100.0 | 0 | 2.22 | 717 | 752 | 48 | 4.0 | yes |
| after: flow 320 | 320 | 1280 | 2 | 60 | 0.934 | 1.095 | 1.11 | 0.0 | 100.0 | 0 | 2.28 | 781 | 822 | 99 | 8.2 | yes |
| after: flow 320 | 320 | 1280 | 4 | 60 | 1.687 | 1.812 | 2.765 | 1.7 | 100.0 | 0 | 3.22 | 717 | 754 | 204 | 17.0 | no |
| after: flow 320 | 320 | 1280 | 8 | 38 | 3.229 | 3.38 | 3.562 | 100.0 | 100.0 | 0 | 3.56 | 781 | 822 | 357 | 29.7 | no |
| after: flow 320, CSRNet 960 | 320 | 960 | 4 | 60 | 1.073 | 1.189 | 1.234 | 0.0 | 98.3 | 4 | 4.06 | 645 | 686 | 155 | 12.9 | yes |
| after: flow 320, CSRNet 960 | 320 | 960 | 8 | 60 | 1.968 | 2.112 | 2.25 | 21.7 | 100.0 | 0 | 2.34 | 518 | 686 | 390 | 32.5 | no |

- NVIDIA GeForce RTX 2050; 12 logical CPUs; clip synthetic_crowd_1280.mp4 (1280x720, 25 fps) for every camera; 120 s per row after 3 warm-up cycles; cycle = app.main.run_cycle every 2 s.
- Latency = time to process every camera once (CSRNet on the GPU, one frame at a time; optical flow on the CPU, cameras in parallel). A cycle is late when it takes longer than the 2 s interval. 'meets_2s_cycle' = p95 latency <= 2 s and at most 1 % of cycles late.
- missed_zone_updates: the test clip is 64 s long and loops; at each wrap a camera's reader drops its buffer (so no burst mixes the clip's end and start) and that camera skips one update, about one miss per camera per 64 s. A live camera does not loop.
- max_update_gap_s = longest time between two updates of one camera (the dashboard shows 'No signal' after 10 s). GPU memory is PyTorch's peak; the CUDA context adds about 0.3-0.5 GB.
- Untrained CSRNet weights: timing is representative (same network), risk levels are not.
- Maximum cameras within the 2 s cycle, before: flow 640: 2
- Maximum cameras within the 2 s cycle, after: flow 320: 2
- Maximum cameras within the 2 s cycle, after: flow 320, CSRNet 960: 4
