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
├── india_dataset/              # python -m tools.india_dataset <step>; see results/INDIA_FINETUNE.md
│   ├── sources.csv             # one row per video: video_file, location, event, year, source_url, camera_type
│   ├── raw_videos/<location>/  # dataset videos (H.264); one folder per location
│   ├── raw_videos/*.mp4        # demo clips (MediaMTX serves each at rtsp://localhost:8554/<name>); not in the dataset
│   ├── frames/<video>/         # ~10 chosen frames, <video>_t0123.4s.jpg + frames.csv, candidates.csv, overlays.json
│   ├── crops/<video>/          # dense crops + contact_sheet.jpg + rejected.txt (crops to leave out)
│   ├── suggestions/<video>/    # suggested head points per crop (pre-annotation)
│   ├── cvat/batch_<n>/         # one CVAT package per annotator (images.zip, preannotations.zip)
│   ├── cvat_exports/           # the annotators' CVAT exports (batch_<n>.zip)
│   ├── crop_annotations/<video>/  # imported, checked head points per crop
│   └── split.csv               # train/val/test by video and location, fixed at the first import
├── eval/                       # counting-accuracy evaluation frames + annotations (never trained on)
├── umn/                        # Crowd-Activity-All.avi + labels.csv (scripts/umn_labels.py) + checks/
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
- **UMN unusual crowd activity** (Phase 6B, turbulence validation). The video is still served by
  the lab (checked 2026-10-13; 24.8 MB, 7739 frames, 30 fps, 320x240), linked from
  http://mha.cs.umn.edu/proj_events.shtml. From the project root:

  ```powershell
  curl.exe -L -o data\umn\Crowd-Activity-All.avi https://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi
  venv\Scripts\python.exe scripts\umn_labels.py        # data\umn\labels.csv from the video's own caption
  venv\Scripts\python.exe scripts\umn_validation.py    # results\umn_validation.md (~10 min, CPU)
  ```

  The ground truth is burnt into the video: a red "Abnormal Crowd Activity" caption from the
  moment the authors mark each of the 11 clips (2 lawn, 6 indoor, 3 plaza) as panicking.
  `umn_labels.py` reads it frame by frame; there is no separate label file to download.
- **India festival videos**: YouTube via `yt-dlp` (see the main README and
  `results/INDIA_FINETUNE.md`), saved as `india_dataset/raw_videos/<location>/<video>.mp4`. Record
  every video in `india_dataset/sources.csv` so the thesis can cite it; check with
  `python -m tools.india_dataset check` (from `backend/`).
