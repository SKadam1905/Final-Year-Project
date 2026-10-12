# Speed surge next to turbulence on UMN (Phase 6B-2)

One burst every 2 s (all 60 offsets within the 2 s evaluated). Each camera is simulated running continuously (3.7 min of normal walking chained per offset), so the smoothing window and the surge history are full when a panic starts. **Main numbers: leave-one-scene-out** (fitted on two scenes, tested on the third). Three alert decisions: the live mean of the last 15 readings within 30 s, each 2 s reading on its own, and two readings in a row. The live pipeline is unchanged.

## Findings

- **Turbulence adds nothing on UMN once the position in the clip is controlled**: time-matched AUC 0.515 with the live smoothing and 0.616 per reading (0.5 = chance). Its plain AUC in results/umn_validation.md partly reflected that every panic ends its clip.
- **Surge ranks panic well in every held-out scene**: time-matched AUC 0.95 / 0.92 / 0.95 (lawn / indoor / plaza) with the live smoothing, 0.99 / 0.72 / 0.73 per reading.
- **But the live 30 s mean blunts it**: a panic lasting 1-3 readings barely moves a 15-reading mean, so at thresholds fitted on the other scenes the smoothed surge flags 0.26 of panic readings and 0.35 of normal ones, and catches 4 / 11 panics. Per reading it flags 0.53 / 0.06 and catches 9 / 11; two readings in a row 0.51 / 0.03 and 9 / 11, with 0.71 false alarms per minute (per reading 1.78).
- **Thresholds**: fitted on the other two scenes, the per-reading surge threshold varied (lawn 1.2, indoor 1.43, plaza 1.2), while the two-in-a-row threshold hardly moved (lawn 1.1, indoor 1.12, plaza 1.1). With two readings in a row, indoor is the hardest scene for surge and for every other signal.
- **Surge vs plain speed**: with two readings in a row speed does about as well here (time-matched AUC 0.866, 0.47 false alarms per minute), but its thresholds are in pixels per frame (lawn 0.871, indoor 0.969, plaza 0.873) and depend on each camera's zoom, resolution and frame rate; UMN's three cameras are alike, so it cannot show the difference. Surge is a ratio to the same camera's recent pace, so its threshold should carry over to other cameras.
- **Combining with turbulence does not help**: the best two-in-a-row method by time-matched AUC is surge (0.896); max(z) and the logistic regression do not beat surge alone, because turbulence carries no panic signal here.

**Recommendation (not applied)**: Do not fold surge into the 30 s mean of the risk pressure: that is where it loses most of its value. Add it as a separate fast trigger next to the smoothed pressure: raise the zone's alert (e.g. to at least ORANGE, through the existing escalation path that bypasses the cooldown) when surge >= 1.10 on two consecutive 2 s readings (speed at least 1.10x its median of the previous 10 s; median delay 1.6 s from the published onset here), and only when the zone's density is above its YELLOW level, so a few people breaking into a run in an empty area cannot raise it. 1.10 is a starting point fitted on UMN, not a deployable setting: it gave 0.71 false alarms per minute of UMN walking (about 43 an hour; normal-walking surge has a long tail), which an operator would not accept; expect to raise it and to rely on the density gate. Keep turbulence in the pressure formula as it is (it does no harm there, but UMN gives no evidence for it), and keep the 0.3 / 0.6 bands. Before any change: validate surge on dense-crowd footage with known surges (UMN is sparse and staged) and measure its false alarms on hours of normal Kumbh footage; 3.7 min of UMN walking cannot pin down a rate of a few per hour.

## Leave-one-scene-out: live 30 s smoothing (as live)

| method | time-matched AUC (lawn / indoor / plaza) | time-matched AUC, mean | plain AUC, mean | TPR / FPR | clips detected | median delay (s) | false alarms / min | thresholds (fitted on the other 2 scenes) |
|---|---|---|---|---|---|---|---|---|
| turbulence | 0.56 / 0.62 / 0.36 | 0.515 | 0.600 | 0.62 / 0.48 | 5 / 11 | 1.0 | 1.70 | lawn 0.163, indoor 0.146, plaza 0.158 |
| speed | 0.94 / 0.82 / 0.90 | 0.885 | 0.894 | 0.27 / 0.34 | 4 / 11 | 1.2 | 0.63 | lawn 0.813, indoor 0.914, plaza 0.821 |
| surge | 0.95 / 0.92 / 0.95 | 0.943 | 0.940 | 0.26 / 0.35 | 4 / 11 | 1.0 | 0.69 | lawn 1.05, indoor 1.17, plaza 1.05 |
| max(z) turbulence, surge | 0.95 / 0.70 / 0.91 | 0.856 | 0.827 | 0.63 / 0.44 | 8 / 11 | 1.0 | 1.45 | lawn 0.458, indoor 0.486, plaza 0.451 |
| logistic turbulence + surge | 0.95 / 0.87 / 0.99 | 0.938 | 0.922 | 0.36 / 0.37 | 6 / 11 | 1.0 | 1.26 | lawn 0.421, indoor 0.133, plaza 0.387 |

## Leave-one-scene-out: each 2 s reading

| method | time-matched AUC (lawn / indoor / plaza) | time-matched AUC, mean | plain AUC, mean | TPR / FPR | clips detected | median delay (s) | false alarms / min | thresholds (fitted on the other 2 scenes) |
|---|---|---|---|---|---|---|---|---|
| turbulence | 0.62 / 0.47 / 0.76 | 0.616 | 0.651 | 0.57 / 0.35 | 9 / 11 | 0.8 | 6.90 | lawn 0.171, indoor 0.155, plaza 0.165 |
| speed | 0.99 / 0.68 / 0.71 | 0.795 | 0.861 | 0.51 / 0.07 | 9 / 11 | 0.9 | 1.86 | lawn 0.95, indoor 1.15, plaza 0.95 |
| surge | 0.99 / 0.72 / 0.73 | 0.812 | 0.849 | 0.53 / 0.06 | 9 / 11 | 1.0 | 1.78 | lawn 1.2, indoor 1.43, plaza 1.2 |
| max(z) turbulence, surge | 0.99 / 0.63 / 0.73 | 0.783 | 0.858 | 0.63 / 0.16 | 10 / 11 | 1.0 | 4.24 | lawn 0.81, indoor 1.01, plaza 1.05 |
| logistic turbulence + surge | 0.99 / 0.79 / 0.73 | 0.835 | 0.869 | 0.54 / 0.06 | 9 / 11 | 1.0 | 1.89 | lawn 0.502, indoor 0.196, plaza 0.489 |
| time into clip (no motion; confound check) | 0.52 / 0.52 / 0.57 | 0.540 | 0.937 | 0.77 / 0.17 | 9 / 11 | 1.0 | 1.87 | lawn 16.2, indoor 17.6, plaza 16.2 |

## Leave-one-scene-out: 2 readings in a row

| method | time-matched AUC (lawn / indoor / plaza) | time-matched AUC, mean | plain AUC, mean | TPR / FPR | clips detected | median delay (s) | false alarms / min | thresholds (fitted on the other 2 scenes) |
|---|---|---|---|---|---|---|---|---|
| turbulence | 0.76 / 0.54 / 0.85 | 0.715 | 0.767 | 0.51 / 0.18 | 8 / 11 | 1.0 | 3.14 | lawn 0.152, indoor 0.153, plaza 0.152 |
| speed | 0.96 / 0.64 / 1.00 | 0.866 | 0.858 | 0.43 / 0.02 | 9 / 11 | 1.7 | 0.47 | lawn 0.871, indoor 0.969, plaza 0.873 |
| surge | 0.96 / 0.73 / 1.00 | 0.896 | 0.853 | 0.51 / 0.03 | 9 / 11 | 1.6 | 0.71 | lawn 1.1, indoor 1.12, plaza 1.1 |
| max(z) turbulence, surge | 0.99 / 0.69 / 1.00 | 0.893 | 0.877 | 0.50 / 0.06 | 8 / 11 | 1.2 | 1.24 | lawn 0.45, indoor 0.782, plaza 0.581 |
| logistic turbulence + surge | 0.92 / 0.74 / 1.00 | 0.886 | 0.849 | 0.43 / 0.05 | 7 / 11 | 1.6 | 1.14 | lawn 0.441, indoor 0.103, plaza 0.431 |

- **Time-matched AUC** is the main ranking measure: each panic reading against normal readings of the same scene at the same time into their clip (within 0.5 s). Plain AUC is shown for comparison: UMN puts every panic at the end of its clip, so it rewards anything that grows over a clip; the last row of the per-reading table, the time into the clip with no motion information, shows how much (plain AUC) and how little is left after matching.
- TPR / FPR: share of panic / normal readings flagged at each held-out scene's threshold. A clip counts as detected when its panic is caught at half or more of the 60 offsets; delay: median over the offsets that catch it, from the published onset; with readings every 2 s about 1 s of it is the wait for the next reading. False alarms: switch-ons per minute in the scenes' chained normal walking.

## Per clip: share of offsets that catch the panic, median delay

Live 30 s smoothing:

| clip | scene | reachable | turbulence | speed | surge | max(z) turbulence, surge | logistic turbulence + surge |
|---|---|---|---|---|---|---|---|
| 1 | lawn | 100% | 0% | 100%, 1.0 s (on before: 48%) | 100%, 1.0 s (on before: 50%) | 100%, 1.0 s (on before: 38%) | 100%, 1.0 s (on before: 50%) |
| 2 | lawn | 100% | 22%, 1.6 s (on before: 13%) | 100%, 1.2 s (on before: 47%) | 100%, 1.0 s (on before: 50%) | 100%, 1.3 s (on before: 42%) | 100%, 1.0 s (on before: 50%) |
| 3 | indoor | 100% | 93%, 0.9 s (on before: 47%) | 0% | 0% | 83%, 0.9 s (on before: 42%) | 28%, 2.5 s (on before: 3%) |
| 4 | indoor | 100% | 32%, 1.3 s (on before: 15%) | 7%, 2.0 s | 17%, 2.1 s | 23%, 1.4 s (on before: 8%) | 77%, 1.5 s |
| 5 | indoor | 100% | 80%, 0.8 s (on before: 37%) | 0% | 0% | 27%, 3.6 s (on before: 8%) | 0% |
| 6 | indoor | 100% | 62%, 1.2 s (on before: 40%) | 7%, 2.0 s | 12%, 2.1 s | 55%, 1.4 s (on before: 23%) | 55%, 1.6 s |
| 7 | indoor | 100% | 100%, 1.0 s (on before: 50%) | 25%, 1.6 s | 0% | 87%, 0.9 s (on before: 28%) | 0% |
| 8 | indoor | 100% | 100%, 1.0 s (on before: 50%) | 2%, 1.3 s | 0% | 90%, 1.1 s (on before: 40%) | 0% |
| 9 | plaza | 65% | 13%, 0.1 s (on before: 8%) | 65%, 0.6 s (on before: 50%) | 65%, 0.6 s (on before: 50%) | 65%, 0.6 s (on before: 50%) | 65%, 0.6 s (on before: 50%) |
| 10 | plaza | 50% | 10%, 0.9 s (on before: 8%) | 50%, 0.5 s (on before: 50%) | 50%, 0.5 s (on before: 50%) | 50%, 0.5 s (on before: 50%) | 50%, 0.5 s (on before: 50%) |
| 11 | plaza | 33% | 22%, 0.3 s (on before: 27%) | 33%, 0.3 s (on before: 50%) | 33%, 0.3 s (on before: 50%) | 33%, 0.3 s (on before: 50%) | 33%, 0.3 s (on before: 50%) |

2 readings in a row:

| clip | scene | reachable | turbulence | speed | surge | max(z) turbulence, surge | logistic turbulence + surge |
|---|---|---|---|---|---|---|---|
| 1 | lawn | 100% | 93%, 0.9 s (on before: 43%) | 100%, 1.6 s | 100%, 1.4 s | 100%, 1.0 s (on before: 33%) | 100%, 1.5 s |
| 2 | lawn | 100% | 92%, 1.2 s (on before: 12%) | 100%, 1.9 s | 100%, 1.8 s (on before: 7%) | 100%, 1.4 s (on before: 3%) | 100%, 1.8 s (on before: 12%) |
| 3 | indoor | 100% | 100%, 1.0 s (on before: 22%) | 92%, 1.7 s | 100%, 1.6 s | 93%, 1.2 s (on before: 8%) | 85%, 2.0 s (on before: 13%) |
| 4 | indoor | 100% | 78%, 1.2 s (on before: 30%) | 63%, 1.9 s | 82%, 1.7 s (on before: 2%) | 82%, 1.5 s (on before: 22%) | 58%, 2.0 s |
| 5 | indoor | 100% | 67%, 1.1 s (on before: 17%) | 53%, 1.3 s | 62%, 1.3 s | 45%, 1.2 s (on before: 12%) | 23%, 1.6 s (on before: 2%) |
| 6 | indoor | 100% | 35%, 1.1 s (on before: 27%) | 43%, 2.0 s | 58%, 1.9 s | 65%, 1.8 s (on before: 8%) | 50%, 2.0 s |
| 7 | indoor | 100% | 97%, 1.7 s (on before: 7%) | 75%, 1.8 s | 83%, 1.8 s | 77%, 1.8 s (on before: 5%) | 38%, 1.9 s |
| 8 | indoor | 100% | 82%, 1.0 s (on before: 23%) | 62%, 1.8 s | 83%, 1.6 s | 77%, 1.4 s (on before: 13%) | 60%, 1.5 s |
| 9 | plaza | 65% | 63%, 0.6 s (on before: 50%) | 65%, 0.6 s (on before: 50%) | 65%, 0.6 s (on before: 50%) | 65%, 0.6 s (on before: 25%) | 65%, 0.6 s (on before: 50%) |
| 10 | plaza | 50% | 47%, 0.5 s (on before: 37%) | 50%, 0.5 s | 48%, 0.5 s | 47%, 0.5 s (on before: 30%) | 47%, 0.5 s |
| 11 | plaza | 33% | 32%, 0.3 s (on before: 23%) | 23%, 0.4 s | 28%, 0.4 s | 22%, 0.4 s (on before: 2%) | 23%, 0.4 s |

Reachable: share of offsets with any reading between the published onset and the end of the caption; no signal can do better (the plaza captions last 0.6-1.3 s). "On before": share of offsets whose alarm was on in the second before the published onset; at most 50%, since a reading falls in that second for half the offsets, so 50% means every such reading was already alarming: people start running before the caption appears.

## Surge values and thresholds refitted on all three scenes

Surge during normal walking: median 1.00, 90th percentile 1.17, 99th 2.90. During panic: 10th percentile 0.82, median 1.49, 90th 2.82.

| method | decision | threshold (all 3 scenes, Youden) |
|---|---|---|
| surge | live 30 s smoothing | 1.046 |
| surge | each 2 s reading | 1.199 |
| surge | 2 readings in a row | 1.098 |
| speed | live 30 s smoothing | 0.816 |
| speed | each 2 s reading | 0.950 |
| speed | 2 readings in a row | 0.871 |
| turbulence | live 30 s smoothing | 0.158 |
| turbulence | each 2 s reading | 0.165 |
| turbulence | 2 readings in a row | 0.152 |

## Method

- **Speed**: the burst's mean speed over moving pixels (>= 0.5 px/frame), full-resolution px/frame: it depends on each camera's zoom, resolution and frame rate.
- **Surge**: speed / median speed of the previous 10 s of readings (5 readings), the median floored at 0.5 px/frame; computed along the continuous stream.
- **max(z)**: max of turbulence and surge after z-scoring each on the training scenes' normal readings. **logistic**: logistic regression (class-balanced) on standardised turbulence and surge from the training scenes. Both are computed per reading, then smoothed, as the live risk pressure is.
- **Labels**: the published caption (results/umn_validation.md); readings spanning the onset or after the caption are left out of AUC / TPR / FPR.
- **Caveat**: 11 clips in 3 scenes, sparse crowds (~20 people) running away, fixed cameras, 30 fps, 320x240; chained normal walking of different clips of one camera stands in for a long continuous recording.
