# CSRNet counting error on ShanghaiTech Part A (test set)

| metric | this_model | paper |
|---|---|---|
| MAE | 71.8 | 68.2 |
| RMSE | 120.2 | 115.0 |

- 182 test images, evaluated once at full resolution; checkpoint csrnet_partA_public.pth, released by the CSRNet authors (their checkpoint choice; nothing tuned here). Images are the processed test set (each side rounded to a multiple of 8 px).
- Paper: Y. Li, X. Zhang, D. Chen, 'CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes', CVPR 2018. Papers in this field label RMSE as 'MSE'.
- Result: this model close to the paper (MAE 5% higher, within 10%).
- Mean true count 434; mean predicted 406 (under-counts on average by 28).
