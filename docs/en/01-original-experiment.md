# 01 · Experiment A: Confusion-Gated Pairwise-Difference Arbitration (Terminated)

[Back to README](../../README.md) · [中文](../zh/01-original-experiment.md)

> **Status**: terminated. Full evaluation never completed; `final_decision = null`. **This is not a falsification.**

## 1. Where the hypothesis came from

A synthetic study (E1–E4 in [`background/`](../../background/)) produced a reproducible effect:

- When the difference signal between two classes is **sparse**, pairwise/relative discrimination crosses the separability threshold about `√D` earlier than global discrimination.
- There is therefore a window in which relative discrimination is already separable while global discrimination is not; introducing pairwise-difference arbitration inside that window should yield a positive gain.
- The gain is **inverted-U shaped**: at very low confusion, global discrimination already suffices (arbitration is pure cost); at very high confusion, neither is separable.

Four pre-registered claims followed:

1. Gain is inverted-U in confusion intensity;
2. A historical-confidence dip appears at the critical band;
3. Trigger cost overtakes the gain at high trigger rates;
4. Online template update degrades arbitration.

## 2. Real-benchmark design

| Item | Setting |
|---|---|
| Datasets | DanceTrack (val, 25 seq), MOT17 (train, 7 seq), MOT20 (train, 4 seq) |
| Protocol | `oracle-detection / association-only`: GT boxes as detections, only the association strategy varies |
| GT filtering | TrackEval `MotChallenge2DBox` convention: keep `conf != 0` and `class == 1` (pedestrian) |
| Features | OSNet (MSMT17 / ImageNet weights), FP32, cross-frame batching, ROCm GPU |
| Associators | `S0_motion` (IoU + linear velocity), `S1_global`, `S2_countgated`, and `S3_frame_calib` / `S3_hist_auc` / `S3_online` / `S3_saturated` / `S3_no_abstain` |
| Evaluation | TrackEval `HOTA / CLEAR / Identity`; official summary read from the `COMBINED` row of `pedestrian_detailed.csv` |
| Statistics | `TAU_GRID` × `ALPHA_GRID` × multiple seeds, bootstrap 95% CI |

The design isolates the detector completely so that the mechanism alone can be judged. The price is `DetA ≈ 0.97`: the conclusions **cannot be extrapolated to real detector output**. That limitation directly motivated Experiment B.

## 3. Evidence actually obtained (all informal)

**Remote full run**: OSNet feature extraction finished for DanceTrack 25/25 and MOT17 7/7. Before `MOT20-04` extraction, the remote instance disconnected for the third time (`V3_SCRIPT_EXIT=1`). Diagnostics, tracking, official evaluation, analysis and the difference report never completed.

**Local CPU dry-run** (5 sequences × 250 frames, [`results/original/full_experiment_report.json`](../../results/original/full_experiment_report.json)):

- `all_conclusions_reproduced = False`;
- `S0_motion` was best; **every `S3` variant was worse**;
- of the four pre-registered claims, only "trigger cost overtakes gain" held; the other three failed or lacked evidence;
- at `alpha = 1.0` (always trigger), IDSW reached 916.

**Informal oracle readouts**: `S3_frame_calib` trigger rate ≈ 1.2%, saturated configuration 100%; feature AUC 0.6807 on DanceTrack and 0.8875 on MOT17.

These numbers consistently point toward "no positive gain under this protocol", but the sample is a subset, not the full run, and the pre-registered formal verdict procedure was never completed. They are recorded as a tendency only.

## 4. A statistical fix that matters

Before the dry-run, a real bug was fixed in `analyze.py`: per-frame ID switches used `prev = cur`, which silently misses switches after occlusions. Corrected to:

```python
for gid, tid in cur.items():
    if gid in prev and prev[gid] != tid:
        switches += 1
prev.update(cur)
```

An `idsw_consistency` quality gate was added; the local dry-run self-computed IDSW `15` matched TrackEval's official `15`. The lesson recurred in Experiment B: after perturbation, the `_key(frame, round(box,1))` box-coordinate identity mapping **silently dropped 63 true-target rows**, so B switched to a `src_gid` side channel.

## 5. Why it stopped, and the final state

Remote instances were recycled/disconnected three times (ports changed repeatedly), and one disconnection lost the extracted `.npy` features before they were mirrored locally. On 2026-11-15 19:02 the author explicitly decided to **stop this experiment and move to a new direction**, without requesting another instance for a full re-run.

Consequently:

- `final_decision` stays `null`;
- the four pre-registered claims are **neither confirmed nor falsified**;
- no optimisation, no new artefact, no protocol change was made;
- these results must not be written up as "the synthetic hypothesis was falsified on real benchmarks".

## 6. Material

- Code: [`code/original/`](../../code/original/) (`exp_config.py`, `data_utils.py`, `extract_features.py`, `tracker.py`, `evaluate.py`, `analyze.py`, `make_diff_report.py`, …)
- Results: [`results/original/`](../../results/original/) (dry-run report, difference report, `partial_results.json` state record, figure)
- Background: [`background/`](../../background/) (synthetic E1–E4 and three figures)
