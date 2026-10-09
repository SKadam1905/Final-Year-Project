# Mean turbulence per motion regime and flow width

| regime | bursts | mean_turbulence_1280 | mean_turbulence_640 | mean_turbulence_480 | mean_turbulence_320 | mean_turbulence_240 |
|---|---|---|---|---|---|---|
| still | 17 | 0.3 | 0.293 | 0.291 | 0.248 | 0.251 |
| orderly | 17 | 0.03 | 0.044 | 0.049 | 0.04 | 0.04 |
| two_lanes | 16 | 0.103 | 0.115 | 0.119 | 0.108 | 0.106 |
| counter_flow | 17 | 0.65 | 0.646 | 0.644 | 0.597 | 0.584 |
| chaotic | 17 | 0.651 | 0.644 | 0.639 | 0.593 | 0.578 |
| half_chaotic | 16 | 0.399 | 0.401 | 0.4 | 0.362 | 0.348 |
| converging | 17 | 0.082 | 0.093 | 0.096 | 0.088 | 0.09 |
| swirl | 16 | 0.107 | 0.119 | 0.123 | 0.118 | 0.123 |

- Sanity check of the metric: still and orderly motion should score near 0, counter-flow and chaotic motion near 1. Regimes are described in scripts/make_test_clip.py.
