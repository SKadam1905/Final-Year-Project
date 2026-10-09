# Ground-truth density maps: sum against head count

| dataset | split | images | heads | density_sum | worst_error_pct | worst_image | maps_over_1pct |
|---|---|---|---|---|---|---|---|
| ShanghaiTech Part A (adaptive kernels) | train | 270 | 146824 | 146824.0 | 2.6e-07 | train_IMG_295 | 0 |
| ShanghaiTech Part A (adaptive kernels) | val | 30 | 15883 | 15883.0 | 1.4e-07 | train_IMG_101 | 0 |
| ShanghaiTech Part A (adaptive kernels) | test | 182 | 78970 | 78970.0 | 2.7e-07 | test_IMG_18 | 0 |
| ShanghaiTech Part B (sigma 15) | train | 360 | 44761 | 44761.0 | 2.3e-07 | train_IMG_145 | 0 |
| ShanghaiTech Part B (sigma 15) | val | 40 | 4519 | 4519.0 | 1.5e-07 | train_IMG_358 | 0 |
| ShanghaiTech Part B (sigma 15) | test | 316 | 39208 | 39208.0 | 2.1e-07 | test_IMG_1 | 0 |

- Each head is a Gaussian normalised to 1 after clipping at the image border, then the map is sum-pooled to 1/8 resolution (equivalent to resizing by 1/8 and multiplying by 64), so a map's sum equals its head count.
- Part A: geometry-adaptive kernels, sigma = 0.3 x mean distance to the 3 nearest heads. Part B: fixed sigma 15.
- Worst relative error over all maps: 2.7e-07% (ShanghaiTech Part A (adaptive kernels) test test_IMG_18). Requirement: within 1%.
- Validation = 30 of Part A's 300 training images (40 of Part B's 400), chosen with a fixed seed; the test images are untouched.
