# CUT3R-rel: all results (as of 2026-10-20)

**Protocol for our runs:**
- **Hardware and precision:** RTX PRO 6000, strict fp32.
- **Streaming:** every frame, in order, one recurrent step per frame.
- **Benchmarks:** KITTI odometry 00–10 (all 11 full sequences, 10 Hz), NRGBD (9 scenes), KITTI depth-val (13 drives).
- **Aggregation:** every number is a mean over all sequences.
- **Paper rows** are copied from Scal3R (arXiv 2609.04201): Table 1, Table 7 and Appendix D.1. They use the same Sim(3) ATE on KITTI 00–10.

**Metrics:**
- **t_err %, r_err (°/100 m):** the KITTI devkit metrics, unaligned.
- **t_err, one scale per sequence:** t_err after a single per-sequence scale.
- **ATE (m):** Sim(3) or SE(3) aligned.
- **RPE over 10 frames:** rotation (°) / translation (m).
- **Forward speed:** predicted ÷ GT forward motion.
- **Depth error:** median-scaled abs. rel. of the self point map.

## KITTI odometry 00–10

| group | run | t_err % | t_err %, one scale per seq | r_err | ATE Sim(3) | ATE SE(3) | RPE10 rot / trans | forward speed | KITTI depth-val depth error |
|---|---|---|---|---|---|---|---|---|---|
| baselines | CUT3R (released) | 70.2 | 68.3 | 24.7 | 210 | 218 | 3.65 / 9.50 | 0.44 | 0.0849 |
| | CUT3R, reported in Scal3R paper | – | – | – | 207.2 | 216.3 | – | – | – |
| | TTT3R (released + TTT3R state update), our run | 63.2 | 62.6 | 22.3 | 183 | 203 | 3.13 / 8.08 | 0.51 | – |
| | TTT3R, reported in Scal3R paper | – | – | – | 182.2 | – | – | – | – |
| | Scal3R + loop closure, paper | – | – | – | 69.7 | 88.7 | – | – | – |
| ours, pilot (2 datasets) | CUT3R-rel v1 (quaternion loss) | 43.9 | 41.8 | 22.8 | 149 | 229 | 3.13 / 1.23 | 1.07 | 0.0877 |
| | CUT3R-rel v2 loss (angle ×10, composed windows) | 44.3 | 41.1 | 23.9 | 147 | 207 | 3.10 / 1.65 | 1.08 | 0.0865 |
| | CUT3R-rel v2 loss + static frames | 46.7 | 44.7 | 26.2 | 156 | 184 | 3.67 / 1.59 | 1.07 | 0.0848 |
| ours, scale-up (5 datasets) | CUT3R-rel, epoch 5 of 10 | 35.0 | 30.8 | 19.9 | 117 | 178 | 2.50 / 1.35 | 1.11 | 0.0839 |
| | **CUT3R-rel, final** | **28.3** | **14.4** | **5.78** | **60.3** | 112 | **0.85** / 2.57 | 1.26 | 0.0813 |
| matched frame-0 controls | pilot frame-0 fine-tune | 68.8 | 67.9 | 17.5 | 217 | 224 | 3.36 / 9.05 | 0.23 | 0.0856 |
| | scale-up frame-0 control | 71.4 | 70.4 | 23.5 | 217 | 225 | 3.27 / 10.3 | 0.31 | 0.0805 |
| state reset every K, chained | released, K = 25 | 12.1 | – | 5.20 | 42.9 | 59.1 | 0.93 / 1.13 | 1.00 | – |
| | released, K = 50 | 13.2 | 11.1 | 4.10 | 47.8 | 70.2 | 1.00 / 1.50 | 1.01 | 0.0874 |
| | released, K = 100 | 24.4 | – | 8.52 | 86.3 | 131 | 1.45 / 2.39 | 1.05 | – |
| | released, K = 200 | 46.3 | – | 16.2 | 127 | 237 | 2.16 / 4.64 | 1.00 | – |
| | pilot frame-0 fine-tune, K = 50 | 19.9 | 18.0 | 7.60 | 105 | 156 | 1.28 / 1.78 | 0.99 | 0.0877 |
| | pilot CUT3R-rel v1, K = 50 | 46.0 | 43.5 | 22.8 | 155 | 230 | 3.06 / 1.31 | 1.07 | 0.0876 |
| | scale-up frame-0, K = 50 | 39.4 | 18.9 | 7.35 | 90.8 | 189 | 1.28 / 3.45 | 1.35 | 0.0818 |
| | scale-up CUT3R-rel, K = 50 | 26.0 | 13.2 | 5.36 | 58.6 | 103 | 0.87 / 2.37 | 1.25 | 0.0810 |
| TTT3R state update | scale-up CUT3R-rel + TTT3R update | 77.1 | 35.2 | 15.5 | 143 | 257 | 2.26 / 9.68 | 2.00 | – |

Scal3R's loop-closure ablation (its Table 7) uses seven sequences: 00, 02, 05, 06, 07, 08 and 09. On that subset:

| run | ATE Sim(3) |
|---|---|
| Scal3R, no loop closure | 143.5 |
| Scal3R + loop closure | 75.0 |
| released CUT3R | 183.7 |
| scale-up CUT3R-rel | 85.5 |
| released, reset every 50 | 56.4 |

## NRGBD (9 scenes)

| group | run | ATE Sim(3) | ATE SE(3) | RPE10 rot / trans | depth error | cross / self point-map error |
|---|---|---|---|---|---|---|
| baselines | CUT3R (released) | 1.18 | 1.56 | 2.17 / 0.140 | 0.0286 | 1.27 |
| | TTT3R (released + TTT3R update) | 0.731 | 0.906 | 0.693 / 0.059 | 0.0262 | 1.22 |
| | Scal3R | not reported | | | | |
| ours, pilot | CUT3R-rel v1 | 1.33 | 2.34 | 2.34 / 0.113 | 0.0297 | 1.00 |
| | v2 loss | 1.24 | 2.50 | 2.10 / 0.132 | 0.0301 | 1.00 |
| | v2 loss + static frames | 1.32 | 2.59 | 3.15 / 0.154 | 0.0293 | 1.01 |
| ours, scale-up | epoch 5 | 1.36 | 1.85 | 2.85 / 0.075 | 0.0299 | 1.01 |
| | **final** | **0.79** | **1.02** | **1.03 / 0.069** | 0.0348 | 1.03 |
| matched frame-0 controls | pilot frame-0 fine-tune | 1.24 | 1.74 | 1.76 / 0.117 | 0.0381 | 1.17 |
| | scale-up frame-0 control | 1.39 | 2.33 | 2.93 / 0.194 | 0.0367 | 1.37 |
| state reset every K, chained | released, K = 50 | 0.394 | 0.50 | 0.435 / 0.037 | 0.0264 | 1.01 |
| | released, K = 100 | 0.432 | 0.546 | 0.516 / 0.042 | 0.0269 | 1.02 |
| | pilot frame-0 fine-tune, K = 50 | 0.571 | 0.704 | 0.487 / 0.042 | 0.0283 | 1.07 |
| | pilot CUT3R-rel v1, K = 50 | 1.30 | 2.07 | 2.18 / 0.107 | 0.0297 | 1.00 |
| | scale-up frame-0, K = 50 | 0.655 | 0.924 | 0.817 / 0.054 | 0.0346 | 1.08 |
| | scale-up CUT3R-rel, K = 50 | 0.80 | 1.07 | 1.05 / 0.070 | 0.0339 | 1.03 |
| TTT3R state update | scale-up CUT3R-rel + TTT3R update | 1.354 | 3.710 | 9.158 / 0.285 | 0.0352 | 1.03 |

"Cross / self point-map error" divides the error of the cross point map (frame 0, or camera t−1 for CUT3R-rel) by that of the self point map. A value of 1.0 means the cross point map has lost nothing to its distance from the origin.

## Diagnostics

**Released CUT3R, frozen** (palindrome streams, `campaigns/frame_anchor`):
- **Range wall:** the predicted camera never gets more than 297.94 m from frame 0. Across 53k positions, GT reaches 2.1 km.
- **Error grows with distance:** the cross/self point-map error ratio goes from 1.0 to 5.1 at 200–400 m from frame 0.
- **Speed collapse:** forward speed falls from 1.0 to 0.2 after about 400 frames.

**Rotation bias, per-frame median (°)**, measured on KITTI 03/04/06/07/10:

| model | preprocessing | yaw bias | roll bias | typical yaw error | t_err % |
|---|---|---|---|---|---|
| pilot CUT3R-rel v1 | default / mirrored / principal-point crop | −0.162 / −0.148 / −0.149 | +0.140 / +0.101 / +0.138 | 0.220 | 40.1 / 35.6 / 37.6 |
| pilot v2 loss | default / mirrored / principal-point crop | +0.161 / +0.201 / +0.174 | +0.181 / +0.147 / +0.186 | 0.217 | 40.1 / 51.2 / 42.3 |
| scale-up frame-0 | default | +0.024 | +0.004 | 0.411 | 70.7 |
| scale-up CUT3R-rel, epoch 5 | default | +0.045 | +0.225 | 0.094 | 31.6 |
| scale-up CUT3R-rel, final | default | +0.021 | +0.019 | 0.073 | 30.8 |

**Constant calibration** of scale-up CUT3R-rel on KITTI (`bias_correct.py`). "Fitted on other 10" means the constants are fitted on the other 10 sequences and applied to the held-out one.

| correction | t_err % | r_err | ATE Sim(3) | ATE SE(3) |
|---|---|---|---|---|
| none | 28.3 | 5.78 | 60.3 | 112 |
| scale, fitted on other 10 | 15.6 | 5.78 | 60.3 | 78.0 |
| rotation offset + scale, fitted on other 10 | 15.1 | 5.42 | 61.3 | 78.1 |
| yaw gain + scale, fitted on other 10 | 11.9 | 4.34 | 59.6 | 71.2 |
| yaw gain + scale, fitted per sequence (upper bound) | 8.5 | 3.28 | 40.4 | 43.9 |

The fitted constants: translation ×1.23 (per sequence 1.15–1.33), yaw ×0.89 (0.86–0.96 on 10 of 11 sequences).

**Field-of-view sweep** of scale-up CUT3R-rel: KITTI cropped narrower so focal ÷ width hits each value (`fov_sweep.py`).

| focal ÷ width | t_err % | t_err %, one scale per seq | ATE Sim(3) | ATE SE(3) | translation gain | yaw gain | depth scale (GT ÷ pred) | depth error |
|---|---|---|---|---|---|---|---|---|
| 0.59 (native KITTI) | 27.4 | 15.0 | 69.9 | 122.1 | 1.215 | 0.903 | 0.905 | 0.0819 |
| 0.68 | 17.7 | 13.9 | 84.1 | 110.7 | 1.096 | 0.961 | 0.958 | 0.0772 |
| 0.79 (nuScenes front camera) | 13.4 | 12.5 | 90.3 | 103.1 | 0.944 | 1.040 | 1.003 | 0.0742 |
| 0.95 | 23.9 | 15.5 | 93.6 | 117.7 | 0.796 | 1.154 | 1.042 | 0.0743 |

**Frame-rate (stride) test** of scale-up CUT3R-rel: every n-th frame. Translation gain = predicted ÷ GT per step. RPE spans 10 kept frames, so it isn't comparable across strides.

| benchmark | stride | t_err % | t_err %, one scale per seq | ATE Sim(3) | ATE SE(3) | translation gain | yaw gain |
|---|---|---|---|---|---|---|---|
| KITTI | 1 (10 Hz) | 28.3 | 14.4 | 60.3 | 112.3 | 1.230 | 0.893 |
| KITTI | 2 | 24.7 | 12.7 | 56.1 | 99.5 | 1.179 | 0.900 |
| KITTI | 3 | 20.8 | 12.4 | 53.1 | 91.4 | 1.125 | 0.906 |
| NRGBD | 1 | – | – | 0.791 | 1.019 | 0.791 | – |
| NRGBD | 3 | – | – | 0.491 | 0.650 | 0.824 | – |
| NRGBD | 10 | – | – | 0.238 | 0.401 | 0.878 | – |

**Depth scale** (GT ÷ prediction, median):

| run | KITTI depth-val | NRGBD |
|---|---|---|
| released | 1.195 | 1.058 |
| pilot v2 loss | 1.148 | 1.046 |
| scale-up frame-0 | 0.907 | 1.031 |
| scale-up CUT3R-rel, epoch 5 | 1.029 | 0.991 |
| scale-up CUT3R-rel, final | 0.886 | 0.999 |

**Training-data audit:** depth/pose reprojection gives 0.999–1.000 in all five training datasets.
