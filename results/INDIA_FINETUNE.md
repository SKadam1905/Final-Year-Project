# Fine-tuning CSRNet on Indian festival footage

The aim is to measure whether a few hundred annotated Indian crowd crops lower the counting error
on Indian crowds, above all at places the model has never seen. The plan's target is an MAE 5–10%
lower than the Part A model alone (Chapter 6). The pipeline below is built and tested; it now
needs only videos and annotations.

**Rules that keep the result honest**

- Split by **video**, never by frame or crop. Frames of one video are near-duplicates, so each
  video goes entirely to train, validation or test.
- Hold out **whole locations** for test: at least 2 locations whose videos are never used in
  training. This measures what matters for Nashik 2027, a model working at a place it has not
  seen.
- The split is chosen once, when the annotations are first imported, and saved in
  `data\india_dataset\split.csv`. Do not move videos between splits after seeing test results.
- The best epoch is chosen on the **validation** videos only.
- Each model (before and after) is evaluated on the test crops **once**:
  `scripts\india_finetune.ps1` does exactly that.

All commands run from `backend\` with the project's virtual environment,
`..\venv\Scripts\python.exe`. Everything under `data\` is git-ignored, because the frames show
faces.

## 1. Collect videos

1. Download clips of dense Indian crowds: Kumbh, Pandharpur, Shirdi, Ganga ghats, temple queues.
   Use H.264 at 720p or more; below 512 px on the short side, crops get smaller than 512 x 512.

   ```powershell
   venv\Scripts\yt-dlp.exe -f "bv*[height<=1080][vcodec^=avc1]" -o "data\india_dataset\raw_videos\nashik_ramkund\ramkund_01.%(ext)s" "<URL>"
   ```
2. Put each video in a **location folder**: `data\india_dataset\raw_videos\<location>\<video>.mp4`.
   Use only letters, digits, `_` and `-`. A video's id is its file name, so ids must be unique
   across locations.
3. Add one row per video to `data\india_dataset\sources.csv`:

   | column | example |
   |---|---|
   | video_file | `nashik_ramkund/ramkund_01.mp4` (path under raw_videos) |
   | location | `nashik_ramkund` (the folder name) |
   | event | `Simhastha Kumbh` |
   | year | `2015` |
   | source_url | the page you downloaded it from (cited in the thesis) |
   | camera_type | `fixed`, `handheld` or `drone` |

   Optional columns are title, channel, upload_date, accessed_date, clip_start_s, clip_end_s
   and notes.
4. Aim for **at least 5 locations**: 2 for test, and the rest for training and validation.
   Several videos per location help too.

The three clips lying directly in `raw_videos\` (not in a folder) are the live demo's. They are
also the source of the counting-accuracy evaluation frames, so they are never part of this
dataset.

## 2. Prepare the crops (about a minute per video on the laptop)

```powershell
..\venv\Scripts\python.exe -m tools.india_dataset check
..\venv\Scripts\python.exe -m tools.india_dataset frames
..\venv\Scripts\python.exe -m tools.india_dataset crops
..\venv\Scripts\python.exe -m tools.india_dataset preannotate
```

1. **check** lists every video with its duration and resolution, and reports videos missing from
   `sources.csv` and rows whose video is missing. It stops on errors (wrong camera type, location
   not matching the folder, duplicate ids). Report: `results\india_intake.md`.
2. **frames** picks about 10 frames per video that differ from each other. It skips frames near
   scene cuts, blurred frames, title cards and blank frames, and frames with little crowd (the
   density model counts each candidate). Frames that show the same people as an already chosen
   frame are skipped too: matched image features on one homography. It also finds static logos
   near the border, which the crops then avoid.
   - Output: `data\india_dataset\frames\<video>\<video>_t0123.4s.jpg`, with the reason for every
     skipped candidate in `candidates.csv`.
   - Report: `results\india_frames.md`.
3. **crops** cuts up to 2 non-overlapping 512 x 512 crops per frame, centred on the densest areas
   the density model sees. Frames are first brought to the live system's resolution
   (`INFER_MAX_WIDTH`). A second crop is cut only if it holds at least 30% of the first one's
   predicted count.
   - Output: `data\india_dataset\crops\<video>\`, with `contact_sheet.jpg`. **Look at every
     contact sheet** and list bad crops (logos, captions, screens, blur, no crowd) in that
     folder's `rejected.txt`.
   - Report: `results\india_crops.md`.
4. **preannotate** suggests head points for every crop. Every person the YOLO detector finds
   gets a head point; the density map, cleared inside the detection boxes, gives points for the
   rest (distant and hidden heads).

Steps 2–4 skip videos already done; add `--force` to redo one (`--video <id>`).

## 3. Annotate in CVAT (the team)

```powershell
..\venv\Scripts\python.exe -m tools.india_dataset batches --annotators "Shubham Kadam,Rohit Kokare,Shraddha Deokar,Mitali Mishra"
```

This deals whole videos to the four annotators, balanced on crops and suggested heads, and writes
one CVAT package per person in `data\india_dataset\cvat\batch_<n>\`. The list of who has which
videos is in `results\india_batches.md`. Each person then follows
**`results\ANNOTATION_GUIDE.md`**:

1. create a CVAT task;
2. upload the suggestions once;
3. correct the points;
4. tag every crop `done` or `bad_crop`;
5. export as *CVAT for images 1.1*.

## 4. Import, check and build the dataset

Put the four export zips in `data\india_dataset\cvat_exports\`, then:

```powershell
..\venv\Scripts\python.exe -m tools.india_dataset import
```

It runs these steps in order:

1. Checks that every dealt crop came back tagged `done` or `bad_crop`. It stops while any crop
   is unfinished; `--allow-incomplete` builds the dataset without those crops.
2. Flags done crops that look like mistakes, in `results\india_import_check.md`:
   - a corrected count far from the suggestion compared with the rest of that video;
   - every suggestion deleted, or every suggestion kept unchanged;
   - double clicks, or points outside the image.
3. Writes the head points to `data\india_dataset\crop_annotations\`.
4. Chooses the split by location, saved in `data\india_dataset\split.csv` and
   `results\india_split.md`: about 15% of the crops for test, from at least 2 locations never
   used in training, and about 15% for validation from whole videos.
5. Builds the density maps (geometry-adaptive Gaussians, as for Part A) in
   `data\processed\india\`.

When locations differ a lot in size, the test share can land far from 15%; `results\india_split.md`
shows the real shares. To name the test and validation videos yourself, add
`--test-videos a,b --val-videos c` (only before any result has been seen).

## 5. Fine-tune and evaluate overnight, in one command

From the project root, with Chrome and other heavy programs closed:

```powershell
Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File scripts\india_finetune.ps1" -WindowStyle Minimized
```

Follow it with `Get-Content results\india_finetune_run.log -Wait -Tail 20`. It runs these steps
in order:

1. `tools.india_dataset build`: rebuilds `data\processed\india\` with the saved split.
2. `ml.evaluate` with the starting weights (`CSRNET_WEIGHTS` in `.env`, or `-Init <file>`) on
   the test crops, written to `results\india_before\`.
3. `ml.train_csrnet --init <starting weights> --lr 1e-6 --epochs 100 --early-stop 20`, with
   1 data worker and no pinned memory (this laptop's RAM). The low learning rate adjusts the Part
   A knowledge rather than overwriting it.
   - The best epoch is chosen on the validation videos and saved to
     `backend\ml\weights\india_ft_best.pth`.
   - The log goes to `results\india_train_log.csv`.
4. `ml.evaluate` with the fine-tuned weights on the same test crops, written to
   `results\india_after\`.
5. `scripts\india_compare.py`: MAE and RMSE before and after, for all test crops, **per test
   location** and per test video.
   - Written to `results\india_finetune.csv` and `results\india_finetune.md`.
   - It also stores the after-numbers in the fine-tuned checkpoint's provenance; check them with
     `..\venv\Scripts\python.exe -m tools.inspect_weights ml\weights\india_ft_best.pth`.

The test error is measured on test **crops**: the densest parts of frames, at the live system's
resolution. It is not measured on whole frames, so say so in the thesis.

## 6. Use the result

If the fine-tuned model has a lower test MAE on the India test locations, set
`CSRNET_WEIGHTS=backend/ml/weights/india_ft_best.pth` in `.env` and restart the backend. The
dashboard footer and `/health` then name these weights. Report all rows of
`results\india_finetune.md` in Chapter 6, whichever model is better, together with:

- the numbers of locations, videos, crops and annotated heads;
- that the test locations were never used in training.

## Ethics

The frames show identifiable faces. Keep `data\` out of git (it already is). Do not publish raw
frames or crops; publish only density maps, head coordinates or blurred examples. Cite every
source video from `sources.csv`.
