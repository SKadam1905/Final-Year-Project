# Measuring a rectangle at Ramkund for zone calibration

`tools/calibrate_zone.py` needs one rectangle on the ground with a known width and length. You click its
four corners in the camera view, and the tool turns that into the zone's true area in m². This guide
covers measuring the rectangle with Google Maps' measure tool.

**First, the camera.** A calibration only holds for the camera view it was made on. You need footage of
Ramkund from a fixed camera: a CCTV feed, or a phone on a tripod that is never moved. The demo clips
are Prayagraj news footage, so a Ramkund measurement does not apply to them.

## 1. Choose the rectangle

- **On the ground people stand on.** Pick it on the same flat level as the zone, such as the paved
  platform. Steps are not one flat surface. If the zone is mostly steps, either outline only the part
  at the rectangle's level, or put the corners on the top and bottom step edges. Either way, list
  the steps as a limitation.
- **Visible in both views.** Each corner must be visible from the camera and recognisable on the
  satellite image. Good corners are paving joints, the rim of a tank or kund, the edges of a platform,
  or the base of a wall. Do not use rooftops: they are higher than the ground and look shifted in
  satellite images.
- **Large.** Make each side at least 15–20 m and place the rectangle near the middle of the zone.
  Google Maps is off by about 0.5 m, which is 3% of 15 m but 10% of 5 m. The tool warns you if the
  zone reaches more than 3 times the rectangle's size away from it.
- **A real rectangle.** Measure both diagonals. Each should be close to √(width² + length²), within
  about 1%. If they differ more than that, the corners are not square: pick another rectangle.

## 2. Measure it on Google Maps (computer)

1. Open maps.google.com, search "Ramkund, Nashik" and choose **Layers → Satellite**. Zoom in as far
   as you can.
2. **Right-click** corner A and choose **Measure distance**. Click corner B. The distance appears in the
   box at the bottom. Check that it is in metres, not feet.
3. Close the box and start a new measurement for each of the other sides (B→C, C→D, D→A) and for both
   diagonals. One measurement per line is easier to read than one long path.
4. **Width** = the average of A→B and C→D. **Length** = the average of B→C and D→A. Opposite sides
   should agree within about 0.5 m. If they do not, a click was off: zoom in and measure again.
5. Record:
   - the date you measured;
   - who measured;
   - what marks each corner, e.g. "NW corner of the platform at the lamp post";
   - a screenshot of each measurement, for the thesis.

**On a phone:**
1. Touch and hold corner A until a red pin appears.
2. Tap the place name at the bottom and choose **Measure distance**.
3. Move the map so the black circle sits on corner B, then tap **Add point**.

**Limits:**
- Satellite images can be years old. Check that the platform looks the same today: work done for the
  2027 Kumbh may have changed it.
- The images can also be off by a metre or so. A tape measure on site is more accurate: if you visit,
  tape the width and one diagonal to check the Google Maps numbers.

## 3. Calibrate

Take a frame where all four corners are visible and not hidden by people (early morning is best).
Then run:

```
cd backend
..\venv\Scripts\python.exe -m tools.calibrate_zone <video or image> --zone RAMKUND_NORTH --method google_maps ^
    --measured-by <name> --measured-on <YYYY-MM-DD> --note "<what marks each corner>"
```

1. Click the corners in order around the rectangle: **A, B, C, D**. A→B must be the side you called
   the width.
2. Type the width, press Enter, then type the length and press Enter.
3. Check that the green ground grid lies flat on the paving. If it does not, press `r` and start
   again.
4. Click the outline of the zone's walkable ground and press Enter.
5. The summary shows the area and how much it would move if your clicks were 1 pixel off. Press `w`
   to save.

Then run `..\venv\Scripts\python.exe -m tools.calibration_report --run` to see how the zone's persons/m²
and risk levels change compared with the old estimate. Stop the demo first: this command runs the
models.
