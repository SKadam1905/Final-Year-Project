"""Pick about 10 distinct, usable frames per video (step 2 of the India dataset pipeline).

One sequential pass over the video collects:
  * scene cuts: HS-histogram Bhattacharyya distance >= 0.3 between consecutive frames (the rule
    the live pipeline uses); candidates within CUT_GUARD_S of a cut are skipped, since
    transitions blend two shots;
  * a candidate every `every_s` seconds, with its sharpness (variance of the Laplacian at 640 px
    width), its share of flat 16x16 blocks (title cards, black frames), a crowd count from the
    density model, ORB features, a thumbnail and a colour histogram;
  * static overlays (channel logos, tickers, CCTV timestamps): pixels near the frame border that
    barely change over most of the video yet have edges. Crops avoid them (ml.crops).
Candidates are rejected as scene cut, title card / blank, blurry or little crowd. A greedy
farthest-point pick then starts from the most crowded frame and adds the frame that looks least
like those already picked, skipping near-duplicates: frames that show the same people (at least
SAME_PEOPLE_INLIERS ORB matches agreeing on one homography, spread over the picture). Features
are taken only where the picture changes during the video, so a fixed camera's static background
does not make all its frames look alike, and not in the corners, where logos sit.

Measured on the two Kumbh news clips: frames 1-3 s apart in one drone shot share 13-100 such
matches; frames from different shots share 0. A handheld ground shot with people walking
keeps fewer (9-13 two seconds apart), hence the threshold of 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

CUT_THRESHOLD = 0.3
CUT_GUARD_S = 0.5
FLAT_BLOCK_STD = 4.0  # grey levels: a 16x16 block this uniform is "flat"
MAX_FLAT_SHARE = 0.75  # more flat blocks than this: title card, black or blank frame
BLUR_RATIO = 0.4  # sharpness below this share of the video's median: motion blur or out of focus
CROWD_RATIO = 0.2  # count below this share of the video's 90th-percentile count: little crowd
MIN_COUNT = 3.0
SAME_PEOPLE_INLIERS = 8
SAME_PEOPLE_SPREAD = 0.05  # the matches' bounding box covers at least this share of the frame
MIN_GAP_S = 3.0
MIN_OVERLAY_FRAMES = 20
WORK_WIDTH = 640  # analysis width for sharpness, flatness and features
_orb = cv2.ORB_create(nfeatures=1000)
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


@dataclass(eq=False)
class Candidate:
    time_s: float
    frame_index: int
    sharpness: float
    flat_share: float
    thumb: np.ndarray = field(repr=False)  # 64x36 grey (float32); background-subtracted after the scan
    hist: np.ndarray = field(repr=False)  # HS histogram
    keypoints: np.ndarray = field(repr=False)  # (N, 2) ORB keypoint positions at WORK_WIDTH
    descriptors: np.ndarray | None = field(repr=False)  # (N, 32) uint8
    near_cut: bool = False
    count: float | None = None
    reason: str = ""  # why it was not picked ("" = picked or usable)


@dataclass
class Scan:
    candidates: list[Candidate]
    cuts_s: list[float]
    overlays: list[tuple[int, int, int, int]]  # x1, y1, x2, y2 in source pixels
    width: int
    height: int
    fps: float
    frames: int


def hs_hist(bgr_small: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist)


def standardise(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    std = x.std()
    return (x - x.mean()) / std if std > 1e-3 else np.zeros_like(x)


def flat_share(grey: np.ndarray, block: int = 16) -> float:
    h, w = (s - s % block for s in grey.shape)
    blocks = grey[:h, :w].reshape(h // block, block, w // block, block).astype(np.float32)
    return float((blocks.std(axis=(1, 3)) < FLAT_BLOCK_STD).mean())


def look_distance(a: Candidate, b: Candidate) -> float:
    """How different two frames look, 0 = alike: colours or layout, whichever differs more."""
    colour = cv2.compareHist(a.hist, b.hist, cv2.HISTCMP_BHATTACHARYYA)
    layout = (1.0 - float((a.thumb * b.thumb).mean())) / 2.0
    return max(colour, layout)


def same_people(a: Candidate, b: Candidate, frame_area: float) -> bool:
    """Do the frames show the same people (enough ORB matches on one homography, spread out)?"""
    if a.descriptors is None or b.descriptors is None or len(a.keypoints) < 8 or len(b.keypoints) < 8:
        return False
    pairs = _matcher.knnMatch(a.descriptors, b.descriptors, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < SAME_PEOPLE_INLIERS:
        return False
    src = a.keypoints[[m.queryIdx for m in good]]
    dst = b.keypoints[[m.trainIdx for m in good]]
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if mask is None or int(mask.sum()) < SAME_PEOPLE_INLIERS:
        return False
    inliers = src[mask.ravel() > 0]
    span = inliers.max(axis=0) - inliers.min(axis=0)
    return float(span[0] * span[1]) >= SAME_PEOPLE_SPREAD * frame_area


def find_overlays(greys: list[np.ndarray], frame_w: int, frame_h: int) -> list[tuple[int, int, int, int]]:
    """Boxes of static overlays near the border (logos, tickers, timestamps), in source pixels.

    A pixel belongs to an overlay when it sits in the outer 20% band, lies on an edge of the
    per-pixel median image, and its median absolute deviation over the video is small. The
    median statistics ignore a minority of frames without the overlay (end cards, interviews).
    Animated or semi-transparent logos change too much to be found; the contact sheet review
    catches crops that contain them.
    """
    if len(greys) < MIN_OVERLAY_FRAMES:
        return []  # too few samples: a pixel can look static by chance
    stack = np.stack(greys).astype(np.float32)
    median = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - median), axis=0)
    h, w = mad.shape
    centre_mad = float(np.median(mad[int(0.3 * h):int(0.7 * h), int(0.3 * w):int(0.7 * w)]))
    if centre_mad < 6:
        return []  # the picture hardly changes, so a static logo cannot be told from static scenery
    band = np.ones((h, w), bool)
    band[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)] = False
    edges = cv2.dilate(cv2.Canny(median.astype(np.uint8), 60, 160), np.ones((3, 3), np.uint8))
    mask = ((mad < max(3.0, 0.25 * centre_mad)) & band & (edges > 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    sx, sy = frame_w / w, frame_h / h
    boxes = []
    for x, y, bw, bh, area in stats[1:]:
        ticker = bh <= 0.12 * h and bw >= 0.5 * w
        if area < 0.001 * h * w or (bw * bh > 0.03 * h * w and not ticker):
            continue  # specks, or large static scenery rather than an overlay
        boxes.append((max(0, round((x - 3) * sx)), max(0, round((y - 3) * sy)),
                      min(frame_w, round((x + bw + 3) * sx)), min(frame_h, round((y + bh + 3) * sy))))
    if sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes) > 0.1 * frame_w * frame_h:
        return []  # overlays are small; this much "static" border is scenery or noise, not logos
    return boxes


def changing_mask(greys: list[np.ndarray]) -> np.ndarray:
    """Where the picture changes over the video (median absolute deviation >= 3 grey levels),
    excluding the four corners (logos); at the greys' resolution."""
    stack = np.stack(greys).astype(np.float32)
    mask = np.median(np.abs(stack - np.median(stack, axis=0)), axis=0) >= 3.0
    h, w = mask.shape
    ch, cw = int(0.15 * h), int(0.2 * w)
    for ys in (slice(0, ch), slice(h - ch, h)):
        for xs in (slice(0, cw), slice(w - cw, w)):
            mask[ys, xs] = False
    return mask


def scan_video(path: Path, count_fn: Callable[[np.ndarray], float] | None = None, every_s: float | None = None,
               max_candidates: int = 120) -> Scan:
    """One pass: scene cuts, candidates with quality metrics, features and counts; static overlays."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    every_s = every_s or max(1.0, total / fps / max_candidates)
    step = max(1, round(every_s * fps))
    candidates, cuts, greys = [], [], []
    prev, i, size = None, 0, (0, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        size = (w, h)
        hist = hs_hist(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA))
        if prev is not None and cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA) >= CUT_THRESHOLD:
            cuts.append(i)
        prev = hist
        if i % step == step // 2:
            grey = cv2.cvtColor(cv2.resize(frame, (WORK_WIDTH, max(1, round(WORK_WIDTH * h / w))),
                                           interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            greys.append(cv2.resize(grey, (WORK_WIDTH // 2, grey.shape[0] // 2), interpolation=cv2.INTER_AREA))
            keypoints, descriptors = _orb.detectAndCompute(grey, None)
            candidates.append(Candidate(
                time_s=i / fps, frame_index=i, sharpness=float(cv2.Laplacian(grey, cv2.CV_64F).var()),
                flat_share=flat_share(grey), hist=hist,
                thumb=cv2.resize(grey, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32),
                keypoints=np.float32([k.pt for k in keypoints]).reshape(-1, 2), descriptors=descriptors,
                count=float(count_fn(frame)) if count_fn else None))
        i += 1
    cap.release()
    guard = CUT_GUARD_S * fps
    for c in candidates:
        c.near_cut = any(abs(c.frame_index - k) <= guard for k in cuts)
    if candidates:
        background = np.median(np.stack([c.thumb for c in candidates]), axis=0)
        changing = changing_mask(greys) if len(greys) >= 8 else None
        for c in candidates:
            c.thumb = standardise(c.thumb - background)  # layout of what moves, not of the static scene
            if changing is not None and len(c.keypoints):
                x = np.clip((c.keypoints[:, 0] / 2).astype(int), 0, changing.shape[1] - 1)
                y = np.clip((c.keypoints[:, 1] / 2).astype(int), 0, changing.shape[0] - 1)
                keep = changing[y, x]
                c.keypoints = c.keypoints[keep]
                c.descriptors = c.descriptors[keep] if c.descriptors is not None and keep.any() else None
    return Scan(candidates, [k / fps for k in cuts], find_overlays(greys, *size), size[0], size[1], fps, i)


def reject_unusable(candidates: list[Candidate]) -> None:
    """Set `reason` on scene-cut, title-card/blank, blurry and little-crowd candidates."""
    for c in candidates:
        if c.near_cut:
            c.reason = "scene cut"
        elif c.flat_share > MAX_FLAT_SHARE:
            c.reason = "title card / blank"
    rest = [c for c in candidates if not c.reason]
    if rest:
        sharp = float(np.median([c.sharpness for c in rest]))
        for c in rest:
            if c.sharpness < BLUR_RATIO * sharp:
                c.reason = "blurry"
    counted = [c.count for c in candidates if not c.reason and c.count is not None]
    if counted:
        floor = max(MIN_COUNT, CROWD_RATIO * float(np.percentile(counted, 90)))
        for c in candidates:
            if not c.reason and c.count is not None and c.count < floor:
                c.reason = "little or no crowd"


def select_frames(candidates: list[Candidate], n: int = 10, frame_area: float | None = None,
                  min_gap_s: float = MIN_GAP_S) -> list[Candidate]:
    """Greedy farthest-point pick among usable candidates, starting from the most crowded;
    candidates showing the same people as a pick, or within min_gap_s of one, are skipped."""
    usable = [c for c in candidates if not c.reason]
    if not usable:
        return []
    area = frame_area or WORK_WIDTH * WORK_WIDTH * 9 / 16
    cache: dict[tuple[int, int], bool] = {}

    def duplicate_of(c: Candidate, picks: list[Candidate]) -> Candidate | None:
        for p in picks:
            if abs(c.time_s - p.time_s) < min_gap_s:
                return p
            key = (id(c), id(p))
            if key not in cache:
                cache[key] = same_people(c, p, area)
            if cache[key]:
                return p
        return None

    picks = [max(usable, key=lambda c: (c.count or 0.0, c.sharpness))]
    while len(picks) < n:
        best, best_d = None, -1.0
        for c in usable:
            if any(c is p for p in picks) or duplicate_of(c, picks) is not None:
                continue
            d = min(look_distance(c, p) for p in picks)
            if d > best_d:
                best, best_d = c, d
        if best is None:
            break
        picks.append(best)
    for c in usable:
        if not any(c is p for p in picks):
            twin = duplicate_of(c, picks)
            c.reason = f"near-duplicate of t={twin.time_s:.1f}s" if twin else "not needed (enough frames)"
    return sorted(picks, key=lambda c: c.time_s)


def read_frame(path: Path, candidate: Candidate) -> np.ndarray:
    """The candidate's full-resolution frame; checks the seek landed on the same picture."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_index)
        ok, frame = cap.read()
        if ok and _same_picture(frame, candidate):
            return frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # inexact seek (some codecs): read forward instead
        for _ in range(candidate.frame_index + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"{path} ended before frame {candidate.frame_index}")
        return frame
    finally:
        cap.release()


def _same_picture(frame: np.ndarray, candidate: Candidate) -> bool:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    return cv2.compareHist(hs_hist(small), candidate.hist, cv2.HISTCMP_BHATTACHARYYA) < 0.02
