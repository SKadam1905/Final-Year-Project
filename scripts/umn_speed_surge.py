"""Phase 6B-2: a "speed surge" signal next to turbulence, evaluated on UMN as the live system would see it.

    venv\\Scripts\\python.exe scripts\\umn_validation.py     # once: per-frame turbulence and speed (results/raw/)
    venv\\Scripts\\python.exe scripts\\umn_speed_surge.py

Signals, from the same 5-frame bursts as scripts/umn_validation.py (production burst_turbulence):
  turbulence   the live score
  speed        the burst's mean speed of moving pixels (full-resolution px/frame)
  surge        speed / median speed of the previous 10 s of readings, the median floored at 0.5 px/frame
               (the live "moving" threshold) so an empty scene cannot divide by ~0. A ratio, so it adapts to
               each camera's normal pace, zoom and frame rate.
  max(z)       max of turbulence and surge, each z-scored on the training scenes' normal walking
  logistic     logistic regression on turbulence and surge (class-balanced, standardised)

Live conditions: one burst every BURST_INTERVAL_S (2 s); the alert decision on the mean of the last
ROLLING_MAXLEN readings within ROLLING_WINDOW_S (15 within 30 s), with app.risk_engine.RollingWindow
itself. Where in each 2 s the burst falls is arbitrary, so all 60 frame offsets are evaluated.

Why streams: every UMN clip ends in its panic and has only 12-26 s of normal walking before it, less
than the 30 s window, so scoring readings within their own clip rewards anything that grows over a clip
(the time into the clip alone reaches AUC 0.93). Here a camera is simulated running continuously:
  * a scene's normal stream chains the normal walking of all its clips, one after another; false alarms
    are counted on it, with the smoothing window and the surge history always full;
  * each clip's panic is played after the normal walking of the scene's other clips (warm-up), so it is
    met with a full window and history; detection delay and the labelled readings come from these.
  * time-matched AUC: each panic reading is compared only with normal readings of the same scene at the
    same time into their clip (within 2 s), so the position in the clip carries no information. The
    time-into-clip baseline is reported to show that (it should be ~0.5).

Main numbers: leave-one-scene-out. Thresholds (Youden's J), z-score parameters and regression weights are
fitted on two scenes and tested on the third. Three alert decisions are compared: the live 30 s mean, each
2 s reading on its own, and two readings in a row (the alarm needs the last two readings above threshold).
Nothing in the live pipeline is changed. Writes results/umn_speed_surge.md (+ .csv tables).
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import get_settings  # noqa: E402
from app.risk_engine import RollingWindow  # noqa: E402
from ml.event_metrics import switch_ons  # noqa: E402
from ml.optical_flow import TurbulenceParams  # noqa: E402

PER_FRAME = ROOT / "results" / "raw" / "umn_turbulence_per_frame.csv"
LABELS = ROOT / "data" / "umn" / "labels.csv"
OUT = ROOT / "results" / "umn_speed_surge"
FPS = 30.0
SCENES = ("lawn", "indoor", "plaza")
HISTORY_S = 10.0
MIN_SPEED = TurbulenceParams().min_speed  # 0.5 px/frame: slower pixels count as static
MATCH_S = 0.5  # time-matched AUC: normal readings within this many seconds of the panic reading's time into its clip
MODES = {"live": "live 30 s smoothing", "single": "each 2 s reading", "two": "2 readings in a row"}
METHODS = ("turbulence", "speed", "surge", "max(z) turbulence, surge", "logistic turbulence + surge")
BASELINE = "time into clip (no motion; confound check)"


def load() -> tuple[list[dict], dict[int, dict]]:
    if not PER_FRAME.exists():
        raise SystemExit(f"{PER_FRAME} not found: run scripts/umn_validation.py first")
    with open(PER_FRAME, newline="", encoding="utf-8") as f:
        frames = [{"clip": int(r["clip"]), "scene": r["scene"], "frame": int(r["frame"]), "label": r["label"],
                   "turbulence": float(r["turbulence"]), "speed": float(r["mean_speed"])} for r in csv.DictReader(f)]
    with open(LABELS, newline="", encoding="utf-8") as f:
        clips = {int(r["clip"]): {k: (int(v) if k not in ("scene", "source") else v) for k, v in r.items()}
                 for r in csv.DictReader(f)}
    return frames, clips


def clip_sequences(frames: list[dict], clips: dict[int, dict], interval: int) -> dict[tuple[int, int], list[dict]]:
    """(clip, offset) -> the readings a live camera would take: one every `interval` frames from `offset`."""
    by_clip: dict[int, dict[int, dict]] = {}
    for r in frames:
        by_clip.setdefault(r["clip"], {})[r["frame"]] = r
    out = {}
    for clip_no, rows in by_clip.items():
        first, last = min(rows), clips[clip_no]["end_frame"]
        for offset in range(interval):
            out[(clip_no, offset)] = [{**rows[f], "t_clip": (f - clips[clip_no]["start_frame"]) / FPS}
                                      for f in range(first + offset, last + 1, interval) if f in rows]
    return out


def with_surge(stream: list[dict], history: int) -> list[dict]:
    """Copies of the readings with `surge` computed along the stream (history carries across clips)."""
    speeds = [r["speed"] for r in stream]
    out = []
    for k, r in enumerate(stream):
        previous = speeds[max(0, k - history):k]
        surge = r["speed"] / max(statistics.median(previous), MIN_SPEED) if previous else 1.0
        out.append({**r, "surge": surge})
    return out


class Scorer:
    """A method's per-reading score, with its parameters fitted on training readings."""

    def __init__(self, method: str, train: list[dict]):
        self.method = method
        if method.startswith("max(z)"):
            normal = [r for r in train if r["label"] == "normal"]
            self.z = {k: (np.mean([r[k] for r in normal]), np.std([r[k] for r in normal])) for k in ("turbulence", "surge")}
        elif method.startswith("logistic"):
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            fit = [r for r in train if r["label"] in ("normal", "panic")]
            x = np.array([[r["turbulence"], r["surge"]] for r in fit])
            self.scaler = StandardScaler().fit(x)
            self.model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(
                self.scaler.transform(x), [r["label"] == "panic" for r in fit])

    def __call__(self, readings: list[dict]) -> np.ndarray:
        m = self.method
        if m in ("turbulence", "speed", "surge"):
            return np.array([r[m] for r in readings], float)
        if m == BASELINE:
            return np.array([r["t_clip"] for r in readings], float)
        if m.startswith("max(z)"):
            return np.maximum(*[(np.array([r[k] for r in readings]) - mu) / sd for k, (mu, sd) in self.z.items()])
        x = np.array([[r["turbulence"], r["surge"]] for r in readings])
        return self.model.predict_proba(self.scaler.transform(x))[:, 1]


def two_in_a_row(scores: np.ndarray) -> np.ndarray:
    """min of each reading and the one before: above a threshold only when both readings are."""
    return np.minimum(scores, np.concatenate([scores[:1], scores[:-1]]))


def smoothed(scores: np.ndarray, interval_s: float, window_s: float, maxlen: int) -> np.ndarray:
    """Live alert smoothing along one stream: mean of the readings in the rolling window (app.risk_engine)."""
    window = RollingWindow(window_s, maxlen)
    out = np.empty_like(scores)
    for k, value in enumerate(scores):
        window.add(k * interval_s, float(value))
        out[k] = window.mean()
    return out


def time_matched_auc(panic: list[tuple[float, float]], normal: list[tuple[float, float]]) -> float:
    """P(panic score > normal score) over pairs whose times into their clips differ by at most MATCH_S."""
    n_t = np.array([t for t, _ in normal])
    n_s = np.array([s for _, s in normal])
    wins = pairs = 0.0
    for t, s in panic:
        near = np.abs(n_t - t) <= MATCH_S
        wins += float((s > n_s[near]).sum() + 0.5 * (s == n_s[near]).sum())
        pairs += float(near.sum())
    return wins / pairs if pairs else float("nan")


def youden(labelled: list[tuple[int, float]]) -> float:
    from sklearn.metrics import roc_curve

    y = np.array([lab for lab, _ in labelled])
    fpr, tpr, thr = roc_curve(y, [s for _, s in labelled])
    return float(thr[int(np.argmax(tpr - fpr))])


class Streams:
    """The continuous-operation streams of the 11 clips at every offset."""

    def __init__(self, sequences, clips, interval, history, interval_s, window_s, maxlen):
        self.clips, self.interval_s, self.window_s, self.maxlen = clips, interval_s, window_s, maxlen
        self.offsets = range(interval)
        self.normal: dict[tuple[str, int], list[dict]] = {}  # (scene, offset) -> chained normal walking
        self.detection: dict[tuple[int, int], tuple[list[dict], int]] = {}  # (clip, offset) -> stream, target start
        for scene in SCENES:
            scene_clips = sorted(c for c in clips if clips[c]["scene"] == scene)
            for o in self.offsets:
                normal = {c: [r for r in sequences[(c, o)] if r["label"] == "normal"] for c in scene_clips}
                self.normal[(scene, o)] = with_surge([r for c in scene_clips for r in normal[c]], history)
                for c in scene_clips:
                    warm = [r for other in scene_clips if other != c for r in normal[other]]
                    target = [r for r in sequences[(c, o)] if r["frame"] <= clips[c]["label_end_frame"]]
                    self.detection[(c, o)] = (with_surge(warm + target, history), len(warm))

    def score(self, stream: list[dict], scorer: Scorer, mode: str) -> np.ndarray:
        s = scorer(stream)
        if mode == "live":
            return smoothed(s, self.interval_s, self.window_s, self.maxlen)
        return two_in_a_row(s) if mode == "two" else s

    def labelled(self, scenes, scorer: Scorer, mode: str) -> list[tuple[dict, float]]:
        """(reading, score) of every clip's own readings in its detection stream (full warm-up behind it)."""
        out = []
        for (c, o), (stream, start) in self.detection.items():
            if self.clips[c]["scene"] in scenes:
                scores = self.score(stream, scorer, mode)
                out += [(r, s) for r, s in zip(stream[start:], scores[start:]) if r["label"] in ("normal", "panic")]
        return out

    def training_readings(self, scenes) -> list[dict]:
        return [r for (c, o), (stream, start) in self.detection.items() if self.clips[c]["scene"] in scenes
                for r in stream[start:]]


def evaluate(streams: Streams, mode: str) -> tuple[list[dict], list[dict], dict]:
    from sklearn.metrics import roc_auc_score

    summary, per_clip, fitted = [], [], {}
    # The time baseline is only meaningful per reading: smoothing a "time into clip" across chained clips is not.
    for method in (*METHODS, BASELINE) if mode == "single" else METHODS:
        row = {"method": method}
        aucs, matched, thresholds = {}, {}, {}
        detected, delays, fa_count, fa_minutes, tp, fp, n_panic, n_normal = 0, [], 0, 0.0, 0, 0, 0, 0
        clip_rows = []
        for test in SCENES:
            train = [s_ for s_ in SCENES if s_ != test]
            scorer = Scorer(method, streams.training_readings(train))
            threshold = youden([(int(r["label"] == "panic"), s) for r, s in streams.labelled(train, scorer, mode)])
            thresholds[test] = threshold
            test_set = streams.labelled([test], scorer, mode)
            y = [int(r["label"] == "panic") for r, _ in test_set]
            x = [s for _, s in test_set]
            aucs[test] = roc_auc_score(y, x)
            matched[test] = time_matched_auc([(r["t_clip"], s) for r, s in test_set if r["label"] == "panic"],
                                             [(r["t_clip"], s) for r, s in test_set if r["label"] == "normal"])
            on = np.array(x) >= threshold
            tp += int(on[np.array(y) == 1].sum())
            fp += int(on[np.array(y) == 0].sum())
            n_panic += sum(y)
            n_normal += len(y) - sum(y)
            for o in streams.offsets:
                stream = streams.normal[(test, o)]
                fa_count += switch_ons(streams.score(stream, scorer, mode) >= threshold)
                fa_minutes += len(stream) * streams.interval_s / 60
            for c in sorted(c for c in streams.clips if streams.clips[c]["scene"] == test):
                onset, last = streams.clips[c]["panic_frame"], streams.clips[c]["label_end_frame"]
                hits, reachable, clip_delays, early = 0, 0, [], 0
                for o in streams.offsets:
                    stream, start = streams.detection[(c, o)]
                    scores = streams.score(stream, scorer, mode)
                    window = [k for k in range(start, len(stream)) if onset <= stream[k]["frame"] <= last]
                    reachable += bool(window)
                    early += bool(any(scores[k] >= threshold for k in range(start, len(stream))
                                      if stream[k]["frame"] < onset and onset - stream[k]["frame"] <= FPS))
                    first = next((k for k in window if scores[k] >= threshold), None)
                    if first is not None:
                        hits += 1
                        clip_delays.append((stream[first]["frame"] - onset) / FPS)
                n = len(streams.offsets)
                detected += hits / n >= 0.5
                if clip_delays:
                    delays.append(statistics.median(clip_delays))
                clip_rows.append({"method": method, "clip": c, "scene": test, "detected": hits / n,
                                  "reachable": reachable / n, "on in the second before": early / n,
                                  "delay": statistics.median(clip_delays) if clip_delays else None})
        row["time-matched AUC (lawn / indoor / plaza)"] = " / ".join(f"{matched[s_]:.2f}" for s_ in SCENES)
        row["time-matched AUC, mean"] = f"{np.mean(list(matched.values())):.3f}"
        row["plain AUC, mean"] = f"{np.mean(list(aucs.values())):.3f}"
        row["TPR / FPR"] = f"{tp / n_panic:.2f} / {fp / n_normal:.2f}"
        row["clips detected"] = f"{detected} / {len(streams.clips)}"
        row["median delay (s)"] = f"{statistics.median(delays):.1f}" if delays else "-"
        row["false alarms / min"] = f"{fa_count / fa_minutes:.2f}"
        row["thresholds (fitted on the other 2 scenes)"] = ", ".join(f"{s_} {thresholds[s_]:.3g}" for s_ in SCENES)
        row["_matched"], row["_thresholds"], row["_detected"] = matched, thresholds, detected
        row["_fa"], row["_tpr_fpr"] = fa_count / fa_minutes, (tp / n_panic, fp / n_normal)
        summary.append(row)
        per_clip += clip_rows
        fitted[method] = thresholds
    return summary, per_clip, fitted


def table(rows: list[dict]) -> list[str]:
    columns = list(rows[0])
    return ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns),
            *("| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows)]


def main() -> None:
    s = get_settings()
    interval = round(s.burst_interval_s * FPS)
    history = round(HISTORY_S / s.burst_interval_s)
    frames, clips = load()
    streams = Streams(clip_sequences(frames, clips, interval), clips, interval, history, s.burst_interval_s,
                      s.rolling_window_s, s.rolling_maxlen)
    results = {mode: evaluate(streams, mode) for mode in MODES}

    proposal = []  # refitted on all three scenes, for the recommendation
    for method in ("surge", "speed", "turbulence"):
        scorer = Scorer(method, streams.training_readings(SCENES))
        for mode, name in MODES.items():
            labelled = streams.labelled(SCENES, scorer, mode)
            proposal.append({"method": method, "decision": name,
                             "threshold (all 3 scenes, Youden)": f"{youden([(int(r['label'] == 'panic'), x) for r, x in labelled]):.3f}"})
    normal_surge = np.array([r["surge"] for st in streams.normal.values() for r in st])
    panic_surge = np.array([r["surge"] for (c, o), (st, start) in streams.detection.items()
                            for r in st[start:] if r["label"] == "panic"])

    def public(rows: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    for stem, data in [(f"_{mode}", public(results[mode][0])) for mode in MODES] + [("_proposal", proposal)]:
        with open(OUT.with_name(OUT.name + stem + ".csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    def pivot(rows: list[dict]) -> list[dict]:
        out = []
        for c in sorted(clips):
            first = next(x for x in rows if x["clip"] == c)
            row = {"clip": c, "scene": clips[c]["scene"], "reachable": f"{first['reachable']:.0%}"}
            for m in METHODS:
                x = next(r for r in rows if r["clip"] == c and r["method"] == m)
                row[m] = (f"{x['detected']:.0%}" + (f", {x['delay']:.1f} s" if x["delay"] is not None else "")
                          + (f" (on before: {x['on in the second before']:.0%})" if x["on in the second before"] else ""))
            out.append(row)
        return out

    normal_minutes = sum(len(st) for (scene, o), st in streams.normal.items() if o == 0) * s.burst_interval_s / 60
    row_of = {mode: {r["method"]: r for r in results[mode][0]} for mode in MODES}
    surge = {mode: row_of[mode]["surge"] for mode in MODES}
    turb_live = row_of["live"]["turbulence"]
    threshold_of = {(r["method"], r["decision"]): float(r["threshold (all 3 scenes, Youden)"]) for r in proposal}
    proposed = threshold_of[("surge", MODES["two"])]
    fa_two = float(surge["two"]["false alarms / min"])
    speed_two = row_of["two"]["speed"]
    two_rows = results["two"][0]
    best_two = max(two_rows, key=lambda r: np.mean(list(r["_matched"].values())))
    worst = {m: min(SCENES, key=lambda sc: r["_matched"][sc]) for m, r in row_of["two"].items()}
    hardest = (f"With two readings in a row, {worst['surge']} is the hardest scene for surge"
               + (" and for every other signal." if len(set(worst.values())) == 1 else
                  "; for the others: " + ", ".join(f"{m} {sc}" for m, sc in worst.items() if m != "surge") + "."))
    findings = [
        f"- **Turbulence adds nothing on UMN once the position in the clip is controlled**: time-matched AUC "
        f"{turb_live['time-matched AUC, mean']} with the live smoothing and "
        f"{row_of['single']['turbulence']['time-matched AUC, mean']} per reading (0.5 = chance). Its plain AUC in "
        "results/umn_validation.md partly reflected that every panic ends its clip.",
        f"- **Surge ranks panic well in every held-out scene**: time-matched AUC "
        f"{surge['live']['time-matched AUC (lawn / indoor / plaza)']} (lawn / indoor / plaza) with the live smoothing, "
        f"{surge['single']['time-matched AUC (lawn / indoor / plaza)']} per reading.",
        f"- **But the live 30 s mean blunts it**: a panic lasting 1-3 readings barely moves a 15-reading mean, so at "
        f"thresholds fitted on the other scenes the smoothed surge flags {surge['live']['TPR / FPR'].split(' / ')[0]} of "
        f"panic readings and {surge['live']['TPR / FPR'].split(' / ')[1]} of normal ones, and catches "
        f"{surge['live']['clips detected']} panics. Per reading it flags {surge['single']['TPR / FPR']} and catches "
        f"{surge['single']['clips detected']}; two readings in a row {surge['two']['TPR / FPR']} and "
        f"{surge['two']['clips detected']}, with {surge['two']['false alarms / min']} false alarms per minute "
        f"(per reading {surge['single']['false alarms / min']}).",
        f"- **Thresholds**: fitted on the other two scenes, the per-reading surge threshold varied "
        f"({surge['single']['thresholds (fitted on the other 2 scenes)']}), while the two-in-a-row threshold hardly "
        f"moved ({surge['two']['thresholds (fitted on the other 2 scenes)']}). " + hardest,
        f"- **Surge vs plain speed**: with two readings in a row speed does about as well here (time-matched AUC "
        f"{speed_two['time-matched AUC, mean']}, {speed_two['false alarms / min']} false alarms per minute), but its "
        f"thresholds are in pixels per frame ({speed_two['thresholds (fitted on the other 2 scenes)']}) and depend on each "
        "camera's zoom, resolution and frame rate; UMN's three cameras are alike, so it cannot show the difference. Surge "
        "is a ratio to the same camera's recent pace, so its threshold should carry over to other cameras.",
        f"- **Combining with turbulence does not help**: the best two-in-a-row method by time-matched AUC is "
        f"{best_two['method']} ({best_two['time-matched AUC, mean']}); max(z) and the logistic regression do not beat "
        "surge alone, because turbulence carries no panic signal here.",
    ]
    recommendation = (
        f"Do not fold surge into the 30 s mean of the risk pressure: that is where it loses most of its value. Add it as a "
        f"separate fast trigger next to the smoothed pressure: raise the zone's alert (e.g. to at least ORANGE, through "
        f"the existing escalation path that bypasses the cooldown) when surge >= {proposed:.2f} on two consecutive "
        f"2 s readings (speed at least {proposed:.2f}x its median of the previous 10 s; median delay "
        f"{surge['two']['median delay (s)']} s from the published onset here), and only when the zone's density is above "
        "its YELLOW level, so a few people breaking into a run in an empty area cannot raise it. "
        f"{proposed:.2f} is a starting point fitted on UMN, not a deployable setting: it gave {fa_two:.2f} false alarms per "
        f"minute of UMN walking (about {fa_two * 60:.0f} an hour; normal-walking surge has a long tail), which "
        "an operator would not accept; expect to raise it and to rely on the density gate. Keep turbulence in the "
        "pressure formula as it is (it does no harm there, but UMN gives no evidence for it), and keep the 0.3 / 0.6 "
        "bands. Before any change: validate surge on dense-crowd footage with known surges (UMN is sparse and staged) "
        "and measure its false alarms on hours of normal Kumbh footage; 3.7 min of UMN walking cannot pin down a "
        "rate of a few per hour."
        if float(surge["two"]["time-matched AUC, mean"]) > float(turb_live["time-matched AUC, mean"]) + 0.1 else
        "The surge signal does not clearly beat turbulence here; do not add it yet.")
    lines = [
        "# Speed surge next to turbulence on UMN (Phase 6B-2)", "",
        f"One burst every {s.burst_interval_s:g} s (all {interval} offsets within the 2 s evaluated). Each camera is "
        f"simulated running continuously ({normal_minutes:.1f} min of normal walking chained per offset), so the "
        "smoothing window and the surge history are full when a panic starts. **Main numbers: leave-one-scene-out** "
        "(fitted on two scenes, tested on the third). Three alert decisions: the live mean of the last "
        f"{s.rolling_maxlen} readings within {s.rolling_window_s:g} s, each 2 s reading on its own, and two readings in "
        "a row. The live pipeline is unchanged.", "",
        "## Findings", "", *findings, "",
        f"**Recommendation (not applied)**: {recommendation}", "",
        *[line for mode, name in MODES.items() for line in
          (f"## Leave-one-scene-out: {name}" + (" (as live)" if mode == "live" else ""), "",
           *table(public(results[mode][0])), "")],
        "- **Time-matched AUC** is the main ranking measure: each panic reading against normal readings of the same "
        f"scene at the same time into their clip (within {MATCH_S:g} s). Plain AUC is shown for comparison: UMN puts every "
        "panic at the end of its clip, so it rewards anything that grows over a clip; the last row of the per-reading "
        "table, the time into the clip with no motion information, shows how much (plain AUC) and how little is left "
        "after matching.",
        "- TPR / FPR: share of panic / normal readings flagged at each held-out scene's threshold. A clip counts as "
        "detected when its panic is caught at half or more of the 60 offsets; delay: median over the offsets that catch "
        "it, from the published onset; with readings every 2 s about 1 s of it is the wait for the next reading. False "
        "alarms: switch-ons per minute in the scenes' chained normal walking.", "",
        "## Per clip: share of offsets that catch the panic, median delay", "",
        *[line for mode in ("live", "two") for line in
          (f"{MODES[mode].capitalize()}:", "", *table(pivot(results[mode][1])), "")],
        "Reachable: share of offsets with any reading between the published onset and the end of the caption; no signal "
        "can do better (the plaza captions last 0.6-1.3 s). \"On before\": share of offsets whose alarm was on in the "
        "second before the published onset; at most 50%, since a reading falls in that second for half the offsets, so "
        "50% means every such reading was already alarming: people start running before the caption appears.", "",
        "## Surge values and thresholds refitted on all three scenes", "",
        f"Surge during normal walking: median {np.median(normal_surge):.2f}, 90th percentile "
        f"{np.percentile(normal_surge, 90):.2f}, 99th {np.percentile(normal_surge, 99):.2f}. During panic: 10th percentile {np.percentile(panic_surge, 10):.2f}, median "
        f"{np.median(panic_surge):.2f}, 90th {np.percentile(panic_surge, 90):.2f}.", "", *table(proposal), "",
        "## Method", "",
        f"- **Speed**: the burst's mean speed over moving pixels (>= {MIN_SPEED} px/frame), full-resolution px/frame: it "
        "depends on each camera's zoom, resolution and frame rate.",
        f"- **Surge**: speed / median speed of the previous {HISTORY_S:g} s of readings ({history} readings), the median "
        f"floored at {MIN_SPEED} px/frame; computed along the continuous stream.",
        "- **max(z)**: max of turbulence and surge after z-scoring each on the training scenes' normal readings. "
        "**logistic**: logistic regression (class-balanced) on standardised turbulence and surge from the training "
        "scenes. Both are computed per reading, then smoothed, as the live risk pressure is.",
        "- **Labels**: the published caption (results/umn_validation.md); readings spanning the onset or after the "
        "caption are left out of AUC / TPR / FPR.",
        "- **Caveat**: 11 clips in 3 scenes, sparse crowds (~20 people) running away, fixed cameras, 30 fps, 320x240; "
        "chained normal walking of different clips of one camera stands in for a long continuous recording.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for mode in MODES:
        for row in results[mode][0]:
            print(mode, "|", row["method"], "|", row["time-matched AUC, mean"], row["plain AUC, mean"], "| detected",
                  row["clips detected"], "| FA/min", row["false alarms / min"])
    print(f"-> {OUT.with_suffix('.md')}")


if __name__ == "__main__":
    main()
