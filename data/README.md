# Data folder

Everything in here except this file and `india_dataset/sources.csv` is git-ignored:
datasets are large, and the India frames contain identifiable faces (plan §5.3: do not
publish raw frames).

```
data/
├── shanghaitech/               # raw ShanghaiTech (verified: Part A 300+182 images, 241,677 heads;
│   ├── part_A_final/           #   Part B 400+316 images, 88,488 heads); any nesting works
│   │   ├── train_data/{images, ground_truth}/   # IMG_1.jpg + GT_IMG_1.mat
│   │   └── test_data/{images, ground_truth}/
│   └── part_B_final/           # same layout
├── shanghaitech_download/      # the downloaded zip (can be deleted after extracting)
├── ucf_qnrf/                   # raw UCF-QNRF (Train/ and Test/ with img_0001.jpg + img_0001_ann.mat)
├── india_dataset/
│   ├── sources.csv             # one row per downloaded video: URL, title, channel, dates (cite these)
│   ├── raw_videos/             # pandharpur_01.mp4, ... (H.264); MediaMTX serves each at rtsp://localhost:8554/<name>
│   ├── frames/<video>/         # frame_0001.jpg ... (tools.extract_frames)
│   └── annotations/<video>/    # frame_0001.json head points (tools.annotate_points / tools.cvat_to_points)
├── umn/                        # Crowd-Activity-All.avi + labels.csv (Phase 5, Part D)
├── test_clips/                 # synthetic_crowd_1280.mp4 (scripts/make_test_clip.py), untrained_csrnet.pth
└── processed/<name>/           # output of ml.prepare_dataset: images/, density/, index.csv
```

## Where to get the datasets

- **ShanghaiTech Part A**: the CSRNet authors' README links it on Google Drive (file id
  `16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI`); it is also mirrored on Kaggle. From the project root:

  ```powershell
  venv\Scripts\pip install gdown
  venv\Scripts\gdown 16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI -O data\shanghaitech_download\
  Expand-Archive data\shanghaitech_download\ShanghaiTech_Crowd_Counting_Dataset.zip data\shanghaitech
  ```

  If Google Drive says the quota is exceeded, open https://github.com/leeyeehoo/CSRNet-pytorch,
  follow its "ShanghaiTech Dataset" link in a browser, and unzip into `data\shanghaitech\`. The
  zip (174 MB) holds Part A and Part B, original images and `GT_IMG_<n>.mat` files. Then build the
  density maps and check them:

  ```powershell
  cd backend
  ..\venv\Scripts\python.exe -m ml.prepare_dataset shanghaitech --workers 2     # Part A, adaptive kernels
  ..\venv\Scripts\python.exe -m ml.prepare_dataset shanghaitech_b --workers 2   # Part B, sigma 15
  cd ..
  venv\Scripts\python.exe scripts\check_density_maps.py   # every map sums to its head count
  ```

  Keep `--workers` at 2 on this 7.6 GB laptop; more workers ran out of memory.

- **UCF-QNRF**: https://www.crcv.ucf.edu/data/ucf-qnrf/ (about 1.5 GB zip).
- **UMN unusual crowd activity** (Part D, turbulence ROC). The lab's page
  (http://mha.cs.umn.edu/proj_events.shtml) no longer offers a download or frame labels, so:
  1. Get the video `Crowd-Activity-All.avi` (11 clips: 2 lawn, 6 indoor, 3 plaza scenes, each
     starting with normal walking and ending with the crowd running away) from a mirror; search
     Kaggle or GitHub for "UMN crowd activity dataset". Put it in `data\umn\`. Note in the thesis
     where it came from.
  2. Copy `scripts\umn_labels_template.csv` to `data\umn\labels.csv` and add one row per clip:
     its first and last frame and the first frame where people start to run (0-based frame
     numbers in the .avi; step through it in VLC or with `tools.extract_frames --fps 25`). If your
     copy comes with published per-frame labels, use those transition frames instead and cite them.
  3. Run `venv\Scripts\python.exe scripts\turbulence_roc.py` from the project root.
- **India festival videos**: YouTube via `yt-dlp` (see the main README). Record every video
  in `india_dataset/sources.csv` so the thesis can cite it.
