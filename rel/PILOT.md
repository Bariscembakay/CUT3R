# CUT3R-rel pilot — record

Question: does predicting each frame relative to the previous one (instead of
frame 0) remove CUT3R's long-stream failures? Motivation and the frozen-model
diagnostic: `lingbot-map/.agents/frame_anchor_diagnostic.md`.

## Setup (2026-10-17)

| | |
|---|---|
| arms | A `config/pilot_frame0.yaml` (CUT3R's own objective) and B `config/pilot_rel.yaml` (`Regr3DPoseRel`); B differs only in the criterion |
| B's objective | pose = camera t -> t-1, cross pointmap in camera t-1, normalisation by camera-frame points, composed-pose terms k = 2..4; no absolute term |
| init | released `cut3r_512_dpt_4_64.pth`, both |
| recipe | stage 4: frozen encoder, 32-view clips, TBPTT chunks of 4, bf16; batch 8, lr 1e-5 cosine to 1e-6, warmup 0.5 epoch, 2 epochs x 2,800 clips = 2,800 optimizer steps |
| data | 1,400 ARKitScenes (300 scenes, `arkitscenes-cut3r-video`) + 1,400 TartanAir v1 clips per epoch; ordered video only, no raymap queries, both arms |
| hardware | 1 x RTX PRO 6000 per arm (`gcp-eu1-rtx6000-kzdd`), 1h45 each; jobs 1114787 / 1114788; wandb `cut3r-rel` runs `syyqnrz2` / `vd4k1c90` |
| code | branch `cut3r-rel`, commit `06c5a10` |
| artifacts | checkpoints `/group/compact-3dmem/campaigns/cut3r_rel/pilot/<arm>/`; evaluation `.../pilot_eval/`, scored by `rel/eval/analyze_pilot.py` -> `pilot_eval/_analysis/` |

Phase-0 checks passed: criterion is 0 on synthetic relative predictions (incl.
across TBPTT chunks) and 3.2 on frame-0 ones; data conventions verified by depth
reprojection (ARKit 0.5%, TartanAir 0.1%); both arms overfit one batch.

## Results

Evaluation: strict fp32, same GPU type and dump code as the released-model
diagnostic. "reset/50" = state reset every 50 frames, chained through one
overlap frame. All scenes of each set.

KITTI odometry 00-10, mean over sequences:

| run | t_err % | r_err deg/100 m | ATE Sim(3) m | RPE@10 rot deg | RPE@10 trans m | fwd speed | max dist m |
|---|---|---|---|---|---|---|---|
| released | 70.2 | 24.7 | 210 | 3.65 | 9.50 | 0.44 | 240 |
| A (frame-0 ft) | 68.8 | 17.5 | 217 | 3.36 | 9.05 | 0.23 | 94 |
| **CUT3R-rel** | **43.9** | 22.8 | 149 | 3.13 | **1.23** | **1.07** | **823** |
| released, reset/50 | 13.2 | 4.1 | 47.8 | 1.00 | 1.50 | 1.01 | 622 |
| A, reset/50 | 19.9 | 7.6 | 105 | 1.28 | 1.78 | 0.99 | 664 |
| CUT3R-rel, reset/50 | 46.0 | 22.8 | 155 | 3.06 | 1.31 | 1.07 | 754 |

Continuous streams by stream position, forward speed ratio / local heading drift (deg per 10 frames):

| position | released | A | CUT3R-rel |
|---|---|---|---|
| 0-50 | 1.01 / 0.61 | 0.92 / 1.15 | 1.04 / 2.38 |
| 100-200 | 0.81 / 3.48 | 0.28 / 2.85 | 1.10 / 2.18 |
| 400-800 | 0.19 / 3.20 | 0.19 / 2.85 | 1.06 / 2.36 |
| >1600 | 0.24 / 3.78 | 0.16 / 3.28 | 1.09 / 2.71 |

NRGBD (9) and KITTI depth-val (13):

| run | NRGBD ATE Sim(3) m | NRGBD RPE@10 rot | NRGBD X_self abs-rel | depth-val X_self abs-rel | depth-val world/self |
|---|---|---|---|---|---|
| released | 1.18 | 2.17 | 0.0286 | 0.0849 | 1.87 |
| A | 1.24 | 1.76 | 0.0381 | 0.0856 | 1.46 |
| CUT3R-rel | 1.33 | 2.34 | 0.0297 | 0.0877 | 1.02 |
| released, reset/50 | 0.394 | 0.435 | 0.0264 | 0.0874 | 1.05 |

### Rotation: precise but biased

Signed per-frame rotation error vs GT on KITTI, median over all frames (deg/frame):

| | bias yaw / pitch / roll | noise (median abs) yaw / pitch / roll |
|---|---|---|
| released | +0.008 / -0.005 / -0.002 | 0.645 / 0.241 / 0.244 |
| A | +0.056 / +0.016 / -0.002 | 0.542 / 0.180 / 0.202 |
| CUT3R-rel | -0.169 / -0.051 / +0.139 | 0.219 / 0.131 / 0.186 |

The yaw bias is negative and the roll bias positive in all 11 sequences.
Removing a constant bias estimated leave-one-out (from the other 10 sequences)
and re-chaining: t_err 43.9 -> 23.4 %, r_err 22.8 -> 11.0 deg/100 m (9 of 11
sequences improve; 01 and 07 get worse).

### Reading

- Versus the matched control A, the relative objective removes the
  translation failures: no progress collapse at any stream position (forward
  speed 1.03-1.10 throughout vs A's 0.16-0.20 after frame 400), 7x lower
  RPE@10 translation, no 298 m wall.
- Its heading drift does not grow with stream position (2.2-2.8 deg / 10
  frames throughout), and a state reset does not help it (46.0 vs 43.9 %):
  consistent with the stationarity argument.
- It does not beat the training-free released + reset/50 (13.2 %). The limit
  is rotation: per-frame noise is 3x lower than released, but a constant bias
  of ~0.17 deg/frame yaw composes into ~170 deg over 1,000 frames. The
  quaternion term gives a near-identity rotation error of 0.17 deg a loss of
  ~0.0015, negligible next to translation, so 2,800 steps did not remove the
  offset the pretrained head carries.
- Fine-tuning hurt the frame-0 model on several axes (forward speed 0.44 ->
  0.23, NRGBD X_self abs-rel 0.0286 -> 0.0381, reset/50 13.2 -> 19.9 %), so
  the matched control was needed: B vs released mixes in that damage.

## Caveats

- One short run per arm (2,800 steps), one seed.
- Leave-one-out debiasing is an estimate of what removing the bias could give,
  not a trained result.
- Training data is synthetic TartanAir plus indoor ARKitScenes; no driving data.

## Rotation bias: where it comes from (2026-10-17)

Signed per-frame rotation error of CUT3R-rel, median, deg/frame (yaw + = right):

| | yaw | pitch | roll |
|---|---|---|---|
| KITTI, all frames | -0.169 | -0.051 | +0.139 |
| KITTI, standing still (< 5 cm/frame) | -0.084 | -0.197 | +0.105 |
| KITTI, first 10 frames of the stream | -0.180 | -0.102 | +0.127 |
| KITTI, GT turning left / straight / right | -0.022 / -0.173 / -0.268 | | +0.05 / +0.15 / +0.18 |
| NRGBD, all frames | -0.006 (5/9 scenes negative) | -0.025 | +0.135 (9/9 positive) |

Released CUT3R has no consistent sign in any of these bins. Training data has no
matching net rotation (TartanAir mean yaw/roll +0.014 / 0.000 deg/step; ARKitScenes,
every 5th scene, +0.14 / +0.06).

- **Roll +0.14 deg/frame is universal** (every KITTI and NRGBD sequence), present
  at zero motion and from the first frame: a constant offset of the relative
  pose head. The quaternion loss near identity is too weak to remove it.
- **Yaw -0.17 deg/frame is driving-specific** (absent indoors): it grows with
  speed and under-rotates right turns (pred yaw ~ 0.9 x GT - 0.17). Candidate
  causes, not yet separated: no driving data in training; KITTI's 512x144
  aspect; a misjudged principal point making forward motion read as yaw.

Smoothness: per-frame rotation noise is 3x lower than released on KITTI (yaw
0.22 vs 0.65 deg) and lower on NRGBD (0.11 vs 0.18); per-step translation
error 7x lower (RPE@10 1.23 vs 9.50 m). Chaining turns per-step error into slow
drift instead of frame-to-frame jitter.

Per-sequence plots: `rel/figures/kitti_xy/`, `rel/figures/nrgbd_topdown/`
(PNG + CSV, unaligned, frame-0 coordinates).
