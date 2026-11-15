# 03 · Experiment B: Explore-Stage Results

[Back to README](../../README.md) · [中文](../zh/03-phase2-explore.md)

> **Status**: explore stage complete (2026-11-15), criterion frozen; **confirm stage never launched**.
> Everything here is a **stage conclusion plus directional tendency**, with `final_decision = null`. Formal H1/H2/H3 verdicts were not completed.

## 1. Scale and completeness

| Item | Value |
|---|---|
| Explore grid | MOT17 5 seq + DanceTrack 5 seq × first 500 frames × 28 frozen conditions × 3 seeds × 6 strategies |
| Output | **5040 cells / 840 groups**, 5071 s (84.5 min) |
| Completeness gate | **11/11 passed**: row count, single hash, stage, seeds, strategies, conditions, benchmarks, sequence count, group count, 6 strategies per group, 8 observables |
| Metric fidelity | The `AssA`/`DetA` implementation vs TrackEval official `HOTA.eval_sequence`: **difference 0** (17/17 self-tests locally and 17/17 on the server) |
| Data | [`results/noise/cells_explore/`](../../results/noise/cells_explore/), 10 per-sequence JSONL shards |

MOT20 was, per pre-registration, **reserved** for leave-one-out external validation and never entered fitting. The frozen config hash was unchanged throughout.

## 2. Primary response

Cluster bootstrap, 1000 resamples, resampling by sequence:

| Quantity | Mean | 95% CI | Reading |
|---|---|---|---|
| **`ΔAssA`** (A4−A0) | **+0.001832** | **[−0.000044, +0.004326]** | **includes 0 → no significant average gain** |
| **`ΔIDSW`** (lower is better) | **+31.43** | **[+25.03, +39.24]** | **significantly positive → significantly more ID switches** |
| `ΔIDF1` | +0.002923 | [+0.000495, +0.005101] | significant but negligible magnitude |

`ΔAssA > 0` in **49.76%** of groups — essentially a coin flip. Only 6 of 28 conditions (c5/c19/c20/c23/c25/c26) have CIs excluding 0, all with magnitude < 0.013.

## 3. The three boundary quantities

Single-factor curves all start from a shared `η=0` baseline. `η*_0` is the first interior extremum of `∂ΔAssA/∂η = 0`; the first sign-change point is reported alongside (both given, no cherry-picking).

| Component | `η*_0` (extremum) | sign change | `η_flip` | `η†` | interior extremum | `ΔAssA` range |
|---|---|---|---|---|---|---|
| `σ_loc` | 0.0338 | 0.0210 | **0** | 0.02 | yes | [−0.0272, +0.0137] |
| `β` (FN) | 0.1206 | 0.0299 | **0** | 0.05 | yes | [−0.0097, +0.0051] |
| `λ` (FP) | 0.0960 | 0.2614 | **0** | 0.05 | yes | [−0.0038, +0.0058] |
| `τ` | — | — | 1.0 | — | no | constant +0.0051 |

Two protocol facts:

- **`η_flip = 0`**: the no-noise baseline already has `ΔIDSW = +0.6`; fine-grained association starts manufacturing ID switches at zero noise. This agrees with the direction of Experiment A, where every `S3` variant was worse under the oracle protocol.
- **`τ` is structurally degenerate as a single factor** (c0 = c16 = c17 bit-identical): τ is the temporal correlation length of failure states; with `σ_loc = β = λ = 0` there are no failures to correlate, so τ has no effect. This is **a structural fact**; τ only acts in the `(τ × β)` interaction conditions (c26/c27).

`η†` is operationalised as the smallest η with `|ΔAssA| < |DetA(η) − DetA(0)|`. All three components satisfy it at very low noise, meaning association differences are quickly swamped by detection degradation.

## 4. Criterion P (frozen)

Model: L2 logistic regression (L2 = 1.0, threshold 0.5, 8 label-free observables, pure numpy for server reproducibility), 840 fitting units, target `sign(ΔAssA)`.

Coefficients on the original scale: `mean_score +5.214`, `p10_score −1.084`, `score_gap +6.374`, `mean_pair_iou −10.609`, `det_per_frame −0.021`, `scale_cv −0.018`, `common_mode_energy −1.060`, `low_score_frac +4.705`; intercept `−3.075`.

**Leave-one-sequence-out diagnostics (inside the explore subset; fitting diagnostics only, not an H2 verdict)**:

| Model | Sign accuracy | 95% CI |
|---|---|---|
| P (8 features) | **0.5512** | [0.5190, 0.5857] |
| Baseline: detection confidence only | **0.6071** | — |
| Baseline: β/λ proxy only | **0.6048** | — |
| Constant majority class | 0.5024 | — |

P's CI lower bound exceeds 0.5, but it **does not beat either simple baseline**. Under the frozen `h2_rule` (require both "CI lower bound > 0.5" and "better than every baseline in all primary leave-one-out folds"), **H2 tends not to hold**.

Frozen file: [`results/noise/frozen_criterion.json`](../../results/noise/frozen_criterion.json), sha256 `47c32ece…8af1`, reproducible by rerunning the analysis.

## 5. Tendencies for the three pre-registered hypotheses (not formal verdicts)

| Hypothesis | Explore-stage signal | Tendency |
|---|---|---|
| H1 (non-monotonic surface + locatable boundary) | all three components have interior extrema; `η*_0` locatable | tends to hold, but effect is tiny, most CIs include 0, weak evidence for a "positive-gain window" |
| H2 (primary: gain sign predictable a priori) | P 0.5512 < baselines 0.6071 / 0.6048 | **tends not to hold** |
| H3 (opposite signs of component derivatives) | all three curves non-monotonic, no clear opposite monotone derivatives | insufficient / tends not to hold |

Per pre-registration, **the explore stage is for modelling only; formal H1/H2/H3 verdicts must be issued once in the confirm stage**. The table is a directional readout, not a conclusion.

## 6. Protocol facts

- Official HOTA matches with `global_alignment_score × IoU`, so **the TP set depends on tracker id assignment**; even with identical injected boxes, `DetA` can differ by ≈ 0.007 across strategies. `A1` (ByteTrack) discards low-score boxes and therefore also changes the detection set, giving a larger `DetA` difference. This is the official definition; `ΔAssA` remains the primary response, and comparisons must state this.
- The confidence pool is uniform per-file sampling over all 21 MOT17 TRAIN `det.txt` files, clipped to [0,1] (156k unique values, std 0.38). MOT20's `det.txt` column 7 is a 0/1 validity flag, not a confidence, so it is excluded.
- `score_gap` is zero in the single-condition smoke test; across the full explore data it has 160 unique values, std 0.0112, and receives a `+6.374` coefficient.
- Real smoke test (MOT17-02-FRCNN / 200 frames / cond18 `σ=0.05, λ=0.2`): all six strategies produced different AssA, `n_amb ≈ 5909`, A4 abstained 21635 times → a measurable signal exists; the grid did not produce a flat zero-difference surface.

## 7. Analysis implementation notes

1. Single-factor curves follow the pre-registered definition, including the `tau == 1` constraint; every curve contains the shared `η=0` baseline.
2. The `frozen_criterion.json` digest seals model parameters only and carries no timestamp, so reruns are identical.

## 8. Not executed (belongs to the confirm stage)

- Full-sequence, full-frame confirm: MOT17 21 + DanceTrack 25 + MOT20 4 = **50 sequences / 50387 frames** (explore was 10 sequences / 5000 frames, ~10× smaller)
- Three leave-one-out validations: leave-one-dataset, leave-one-detector (primary); leave-one-noise-family (secondary)
- Official `det.txt` anchor protocol for external validity, reported alongside the injection protocol
- Formal H1/H2/H3 verdicts under Holm correction, plus calibration curves
- A5's RQ3 attribution

Cost estimate: with the frozen `MAIN_RESPONSE = (A4, A0)` only, ≈ 8400 cells, extrapolating from observed explore throughput ≈ **5–8 GPU hours**.

Full report: [`design/PHASE2_EXPLORE_STAGE_REPORT.en.md`](../../design/PHASE2_EXPLORE_STAGE_REPORT.en.md) ([Chinese original](../../design/PHASE2_EXPLORE_STAGE_REPORT.md)); machine-readable summary [`results/noise/explore_summary.json`](../../results/noise/explore_summary.json).
