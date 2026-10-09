# Fine-tuning CSRNet on Indian festival footage

The aim is to measure whether a few hundred annotated Indian frames lower the counting error on
Indian crowds. The plan's target is a 5–10% lower MAE than the Part A model alone (Chapter 6).
Everything below is ready to run; it waits only for annotated frames.

**Rules that keep the result honest**

- Split by **video**, never by frame. Frames of one video are near-duplicates, so a video's
  frames go entirely to train, validation or test. `ml.prepare_dataset` refuses a video
  listed in both test and validation.
- Choose the test videos **before** fine-tuning, and do not change them after seeing results.
- The best epoch is chosen on the **validation** videos only.
- Each model (before and after) is evaluated on the test videos **once**.
  `scripts\india_finetune.ps1` does exactly that.

## 1. Collect videos

1. Download clips of dense Indian crowds, ideally Kumbh, Pandharpur, Shirdi or Ganga ghats, from
   a fixed or slowly moving camera. Use H.264, up to 720p:

   ```powershell
   venv\Scripts\yt-dlp.exe -f "bv*[height<=720][vcodec^=avc1]" -o "data\india_dataset\raw_videos\pandharpur_01.%(ext)s" "<URL>"
   ```
2. Record every clip in `data\india_dataset\sources.csv`: id, URL, title, channel, upload date
   and download date. You will need these for the thesis citations.
3. Aim for at least 6 videos: at least 2 for test, 1 for validation, and the rest for training.
   Vary the places and viewing angles; a model tested on a scene it was trained on proves little.

## 2. Extract frames

One frame every 5 seconds avoids near-duplicates within a video:

```powershell
cd backend
..\venv\Scripts\python.exe -m tools.extract_frames ..\data\india_dataset\raw_videos --fps 0.2
```

Frames go to `data\india_dataset\frames\<video>\frame_0001.jpg`, and so on. Around 150–300 frames
in total is a realistic amount to annotate. Delete frames that are blurred, show captions only,
or show no crowd.

## 3. Annotate heads in CVAT

CVAT is free at https://app.cvat.ai (create an account; nothing to install).

1. **Project:** Projects → + → name `kumbh-heads`. Add one label named `head`, type **Points**.
2. **One task per video:** Tasks → + → name it exactly by the video id (e.g. `pandharpur_01`),
   project `kumbh-heads`. Upload all frames of `data\india_dataset\frames\pandharpur_01\`.
   Leave image quality at 100.
3. **Annotate:** open the job. Choose the **Draw new points** tool (the point icon), label
   `head`, then click once on the centre of every visible head. Press **N** to start and finish a
   set of points, and **Ctrl+S** to save often.
   - Include partly hidden heads and heads far away. Zoom in (mouse wheel) on distant areas.
   - Exclude people on posters or screens, reflections in water, and statues.
   - One click per head, at the head (not the body), the same way in every frame.
   - A dense frame takes 10–20 minutes; use **Ctrl+Z** to undo a stray click.
4. **Export:** Tasks → ⋮ → **Export task dataset** → format **CVAT for images 1.1** →
   download and unzip. You get an `annotations.xml`.

(Alternative without CVAT: the built-in click tool,
`..\venv\Scripts\python.exe -m tools.annotate_points ..\data\india_dataset\frames\pandharpur_01`.)

## 4. Import the annotations

For each video's export, from `backend\`:

```powershell
..\venv\Scripts\python.exe -m tools.cvat_to_points C:\path\to\annotations.xml --video pandharpur_01 --label head
```

This writes `data\india_dataset\annotations\pandharpur_01\frame_0001.json` (head points) for
every annotated frame and prints `<n> frames, <m> heads`. It warns if a frame in the export is not
under `data\india_dataset\frames\<video>\`, where the next step looks for it. Frames without an
annotation file are left out of the dataset.

## 5. Fine-tune and evaluate, in one command

From the project root, name the test and validation videos:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\india_finetune.ps1 -TestVideos shirdi_01,ghat_03 -ValVideos pandharpur_02
```

It runs, in order:

1. `ml.prepare_dataset india`: density maps from the head points (geometry-adaptive Gaussians,
   as for Part A) and the split by video, written to `data\processed\india\`.
2. `ml.evaluate` with the starting weights (`CSRNET_WEIGHTS` in `.env`, or `-Init <file>`) on the
   test videos, written to `results\india_before\`.
3. `ml.train_csrnet --init <starting weights> --lr 1e-6 --epochs 60`: a low learning rate so the
   Part A knowledge is adjusted, not overwritten. The best epoch is chosen on the validation
   videos and saved to `backend\ml\weights\india_ft_best.pth`; the log goes to
   `results\india_train_log.csv`.
4. `ml.evaluate` with the fine-tuned weights on the same test videos, written to `results\india_after\`.
5. `scripts\india_compare.py`: MAE and RMSE before and after, overall and per test video, written
   to `results\india_finetune.csv` and `results\india_finetune.md`. It also stores the
   after-numbers in the fine-tuned checkpoint's provenance. Check it with
   `..\venv\Scripts\python.exe -m tools.inspect_weights ml\weights\india_ft_best.pth`.

The training log prints the seconds per epoch and an estimated finish time as it runs.

## 6. Use the result

If the fine-tuned model has a lower test MAE on the India videos, set
`CSRNET_WEIGHTS=backend/ml/weights/india_ft_best.pth` in `.env` and restart the backend. The
dashboard footer and `/health` then name these weights. Report both rows of
`results\india_finetune.md` in Chapter 6, whichever is better, together with the number of
annotated frames and heads.

## Ethics

The frames show identifiable faces. Keep `data\` out of git (it already is), do not publish raw
frames, publish only density maps or head coordinates, and cite every source video from
`sources.csv`.
