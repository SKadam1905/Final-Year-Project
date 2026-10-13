# Annotation guide: marking heads in CVAT

For the four annotators of the Indian crowd dataset. Each of you gets one **batch**: a set of
crowd crops (square pieces of video frames). The computer has already put a yellow point on
every head it thinks it sees. Your job is to **correct those points**: move them onto heads,
delete the wrong ones and add the missing ones. Then tag each crop as finished.

The crops show real people's faces. Do not share them outside the team or post them anywhere,
and delete your local copies when the project is over.

## Who gets which batch

The coordinator deals the batches with one command (see *For the coordinator* at the end). The
list is then in `results\india_batches.md`. Each batch holds whole videos, so no video is split
between two people, and the batches are balanced on the number of crops and suggested heads.

| Batch | Annotator | Package folder |
|---|---|---|
| batch_1 | Shubham Kadam | `data\india_dataset\cvat\batch_1\` |
| batch_2 | Rohit Kokare | `data\india_dataset\cvat\batch_2\` |
| batch_3 | Shraddha Deokar | `data\india_dataset\cvat\batch_3\` |
| batch_4 | Mitali Mishra | `data\india_dataset\cvat\batch_4\` |

`results\india_batches.md` lists the videos, the number of crops and the suggested heads in each
batch. Suggested heads undercount dense crowds (often by several times in drone shots), so a batch
of dense aerial crops is more work than its number suggests.

Your package folder holds:

- `images.zip`: your crops (JPEG, at most 512 x 512 pixels);
- `preannotations.zip`: the suggested points, in CVAT's format;
- `annotations.xml`: the same points unzipped (you need only one of the two).

## 1. Create your task in CVAT (once)

1. Go to <https://app.cvat.ai> and sign in (a free account is enough). Each person works in
   their own account.
2. Go to **Tasks** and click **+**, then **Create a new task**.
3. Set **Name** to your batch, e.g. `india_batch_2`.
4. Under **Labels**, open the **Raw** tab, replace its contents with the text below, and click
   **Done**:

   ```json
   [
    {"name": "head", "type": "points", "attributes": []},
    {"name": "done", "type": "tag", "attributes": []},
    {"name": "bad_crop", "type": "tag", "attributes": []}
   ]
   ```

   The same text is in `data\india_dataset\cvat\labels.json`. The names must be exactly `head`,
   `done` and `bad_crop`, or the import step later will not find your work.
5. Under **Select files**, choose **My computer** and add your `images.zip`.
6. Open **Advanced configuration** and set **Image quality** to **100**, so that small, distant
   heads stay sharp. Leave the sorting at **Lexicographical**.
7. Click **Submit & Open**.

## 2. Load the suggested points (once, before you start)

1. On the task page, open **Actions**, then **Upload annotations**.
2. Choose the format **CVAT 1.1** (some versions call it *CVAT for images 1.1*) and select your
   `preannotations.zip`. If it is not accepted, select `annotations.xml` instead.
3. Open the job. Every crop should show yellow points.

**Upload the suggestions only once, right after creating the task.** CVAT removes all existing
annotations when you upload, so uploading again later would wipe out your corrections.

## 3. Correct the points

Open the job from the task page. Work crop by crop: **F** goes to the next image and **D** to the
previous one. Zoom with the mouse wheel and save often with **Ctrl+S**.

**The rules: one point per head, the same way on every crop**

1. Click the **centre of the head**, not the face, hat or body. For someone seen from behind or
   from above, it is still the centre of the head.
2. Mark every head you can see: partly hidden heads, heads far away and heads in the water
   included. A head cut off by the crop's edge counts only if its centre is inside the crop.
3. **If you are not sure it is a head, skip it.** Do not guess in areas that are only coloured
   blur. A missing head costs less than an invented one.
4. **Never mark** people on posters, banners, hoardings, TV or phone screens, reflections in water
   or glass, statues, idols or pictures. Do not mark anything under a channel logo or caption.
5. One point per head: no duplicates. Two points on the same head look like a double click to
   the import check.

**Editing**

- **Move a point**: drag it onto the head.
- **Delete a suggested point**: each suggestion is its own object. Hover over it so it is
  highlighted, then press **Delete**.
- **Add heads**: pick the **Points** tool in the left toolbar, choose the label `head`, click
  **Shape**, then click on each missing head. Press **N** (or **Done** on the top bar) to
  finish. One shape can hold many points.
- **Delete one point of a shape you added**: **Ctrl+click** it, or right-click it and choose
  **Delete point**.
- **Undo**: **Ctrl+Z**.

**How to work through a crop**

- **Sparse crops** (a few large people): the suggestions are usually close. Check each one, then
  look for missing people.
- **Dense crops** (a packed crowd, drone shots): the suggestions show where the model sees
  people, but they sit on a rough grid and are too few. Work in strips, left to right and top to
  bottom, zoomed in. Move the points that are near a head and delete the rest. If almost none are
  usable, deleting them and clicking fresh is often faster. Take a break every 20–30 minutes:
  accuracy drops when you are tired.

## 4. Tag every crop: `done` or `bad_crop`

When a crop is finished, give it the tag **`done`**. If a crop cannot be annotated, give it
**`bad_crop`** instead, and do not spend time on its points. A crop is bad when:

- it is mostly a logo, caption, screen or banner;
- it is blurred, so heads cannot be told apart;
- it shows no crowd;
- it is the same view as another crop you already did.

To tag:

1. Switch the workspace (the drop-down in the top-right corner, normally *Standard*) to
   **Tag annotation**.
2. Pick the label `done` (or `bad_crop`) and click the **+** button.
3. Optionally, tick **Automatically go to the next frame** while you tag a run of finished crops.
4. Switch back to **Standard** to keep correcting points.

The import refuses a batch while any crop has neither tag. This is how we know every crop was
checked.

## 5. Export and hand in

1. Save (**Ctrl+S**). On the **Tasks** page, open your task's **⋮** menu, then **Export task
   dataset**.
2. Choose the format **CVAT for images 1.1**. Leave **Save images** off: only the points are
   needed, and the file stays small.
3. Download the zip and rename it after your batch, e.g. `batch_2.zip`.
4. Send it to the coordinator, who puts it in `data\india_dataset\cvat_exports\`.

You can export again at any time. The latest export replaces the earlier one when the
coordinator imports.

## 6. After the import check

The coordinator's import writes `results\india_import_check.md`. It lists crops that are not
tagged yet, and done crops that look like mistakes:

- **"far from the usual change for this video"**: the corrected count differs from the suggestion
  much more than on your other crops of the same video. Re-check it; if the count is right,
  nothing needs to change.
- **"all suggestions kept exactly as they were"**: was the crop really checked?
- **"all suggestions deleted and no head added"**: it should probably be `bad_crop`.
- **"pairs of points less than 2 px apart"**: double clicks; delete one of each pair.
- **"points outside the image"**: delete them.

Fix the listed crops in CVAT, export again, and hand in the new zip.

---

## For the coordinator

From `backend\` (always the project's virtual environment, `..\venv\Scripts\python.exe`):

```powershell
..\venv\Scripts\python.exe -m tools.india_dataset check
..\venv\Scripts\python.exe -m tools.india_dataset frames
..\venv\Scripts\python.exe -m tools.india_dataset crops
#   review data\india_dataset\crops\<video>\contact_sheet.jpg; list bad crops in that folder's rejected.txt
..\venv\Scripts\python.exe -m tools.india_dataset preannotate
..\venv\Scripts\python.exe -m tools.india_dataset batches --annotators "Shubham Kadam,Rohit Kokare,Shraddha Deokar,Mitali Mishra"
#   give each person their data\india_dataset\cvat\batch_<n>\ folder and this guide
#   ...annotation in CVAT...
#   put the four exports in data\india_dataset\cvat_exports\, then:
..\venv\Scripts\python.exe -m tools.india_dataset import
```

Batches are dealt only once: `batches` refuses to deal again, because people work from the
packages. Use `--force` only if nobody has started. A crop listed in `rejected.txt` after
dealing is still in someone's batch; have them tag it `bad_crop`.
