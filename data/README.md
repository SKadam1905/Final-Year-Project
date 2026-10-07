# Data folder

Everything in here except this file and `india_dataset/sources.csv` is git-ignored:
datasets are large, and the India frames contain identifiable faces (plan §5.3: do not
publish raw frames).

```
data/
├── shanghaitech_part_a/        # raw ShanghaiTech Part A (482 images, 241,677 heads)
│   └── part_A_final/           #   any nesting works; the tool finds train_data/ and test_data/
│       ├── train_data/{images, ground-truth}/   # IMG_1.jpg + GT_IMG_1.mat
│       └── test_data/{images, ground-truth}/
├── ucf_qnrf/                   # raw UCF-QNRF (Train/ and Test/ with img_0001.jpg + img_0001_ann.mat)
├── india_dataset/
│   ├── sources.csv             # one row per downloaded video: URL, title, channel, dates (cite these)
│   ├── raw_videos/             # pandharpur_01.mp4, ... (H.264); MediaMTX serves each at rtsp://localhost:8554/<name>
│   ├── frames/<video>/         # frame_0001.jpg ... (tools.extract_frames)
│   └── annotations/<video>/    # frame_0001.json head points (tools.annotate_points / tools.cvat_to_points)
└── processed/<name>/           # output of ml.prepare_dataset: images/, density/, index.csv
```

## Where to get the datasets

- **ShanghaiTech Part A**: linked from the CSRNet authors' repository
  (https://github.com/leeyeehoo/CSRNet-pytorch, "Datasets" section) and mirrored on Kaggle.
  Only Part A is used for training.
- **UCF-QNRF**: https://www.crcv.ucf.edu/data/ucf-qnrf/ (about 1.5 GB zip).
- **UMN unusual crowd activity** (turbulence ROC curve): http://mha.cs.umn.edu/proj_events.shtml
- **India festival videos**: YouTube via `yt-dlp` (see the main README). Record every video
  in `india_dataset/sources.csv` so the thesis can cite it.
