# Phase 2 Explore Subset · Stage Report

**English counterpart of [`PHASE2_EXPLORE_STAGE_REPORT.md`](PHASE2_EXPLORE_STAGE_REPORT.md)**
Generated 2026-11-15 · **Status: explore complete, criterion frozen; confirm stage NOT launched; `final_decision = null`**

- Frozen config: `results/noise/frozen_config.json` r4, sha256 `71dc864e…4be63` (unchanged throughout)
- Frozen criterion: `results/noise/frozen_criterion.json`, sha256 `47c32ece…8af1` (reproducible, rerun digest identical)

---

## 1. One-sentence conclusion

Across 5040 controlled units, **the marginal association gain of fine-grained association (cross-agreement abstention `A4`) over the IoU baseline `A0` is near zero and statistically insignificant, while ID switches increase significantly; the label-free criterion `P` designed to predict the sign of that gain before evaluation does not beat two simple baselines**. Per the pre-registered design this points toward a corrective negative conclusion; the formal verdict was never completed.

## 2. Execution and completeness

| Item | Result |
|---|---|
| Explore scale | 10 sequences (MOT17 5 + DanceTrack 5) × first 500 frames × 28 frozen conditions × 3 seeds × 6 strategies |
| Output | **5040 cells / 840 groups**, 5071 s (≈ 84.5 min) |
| Completeness gate | **11/11 passed**: row count, single hash, stage, seeds, strategies, conditions, benchmarks, sequence count, group count, 6 strategies per group, 8 observables |
| Metric fidelity | The `AssA`/`DetA` implementation vs TrackEval official `HOTA.eval_sequence`: **difference 0** (17/17 self-tests locally and on the server) |

MOT20 was, per pre-registration, **entirely reserved** for leave-one-out external validation in the confirm stage and never entered fitting.

## 3. Primary response (cluster bootstrap, 1000 resamples, by sequence)

| Quantity | Mean | 95% CI | Reading |
|---|---|---|---|
| **`ΔAssA`** (A4−A0) | **+0.001832** | **[−0.000044, +0.004326]** | **includes 0 → no significant average gain** |
| **`ΔIDSW`** (lower is better) | **+31.43** | **[+25.03, +39.24]** | **significantly positive → significantly more ID switches** |
| `ΔIDF1` | +0.002923 | [+0.000495, +0.005101] | significant but negligible magnitude |

`ΔAssA > 0` in **49.76%** of groups, essentially a coin flip. Only 6 of 28 conditions (c5/c19/c20/c23/c25/c26) have CIs excluding 0, all with magnitude < 0.013.

## 4. The three boundary quantities (per frozen definitions)

Single-factor curves share the `η=0` baseline; `η*_0` is the first interior extremum of `∂ΔAssA/∂η=0`, and the first sign-change point is reported alongside (both given, no cherry-picking).

| Component | `η*_0` (extremum) | sign change | `η_flip` | `η†` | interior extremum | `ΔAssA` range |
|---|---|---|---|---|---|---|
| `σ_loc` | 0.0338 | 0.0210 | **0** | 0.02 | yes | [−0.0272, +0.0137] |
| `β` (FN) | 0.1206 | 0.0299 | **0** | 0.05 | yes | [−0.0097, +0.0051] |
| `λ` (FP) | 0.0960 | 0.2614 | **0** | 0.05 | yes | [−0.0038, +0.0058] |
| `τ` | — | — | 1.0 | — | no | constant +0.0051 |

Two protocol facts:

- **`η_flip = 0`**: the no-noise baseline already has `ΔIDSW = +0.6`. Fine-grained association starts manufacturing ID switches at zero noise, consistent with Experiment A's oracle result where every `S3` variant was worse.
- **`τ` is structurally degenerate as a single factor** (c0 = c16 = c17 bit-identical): τ is the temporal correlation length of failure states, and with `σ_loc = β = λ = 0` there are no failures to correlate. This is **a structural fact, not a bug**; τ acts only in the `(τ × β)` interaction conditions (c26/c27).

`η†` is the smallest η with `|ΔAssA| < |DetA(η) − DetA(0)|`. All three components satisfy it at very low noise: association differences are quickly swamped by detection degradation.

## 5. Criterion P (frozen)

L2 logistic regression (L2 = 1.0, threshold 0.5, 8 label-free observables, pure numpy for server reproducibility), 840 fitting units, target `sign(ΔAssA)`.

Coefficients on the original scale: `mean_score +5.214`, `p10_score −1.084`, `score_gap +6.374`, `mean_pair_iou −10.609`, `det_per_frame −0.021`, `scale_cv −0.018`, `common_mode_energy −1.060`, `low_score_frac +4.705`; intercept `−3.075`.

**Leave-one-sequence-out diagnostics (inside the explore subset; fitting diagnostics only, NOT an H2 verdict)**

| Model | Sign accuracy | 95% CI |
|---|---|---|
| P (8 features) | **0.5512** | [0.5190, 0.5857] |
| Baseline: confidence only | **0.6071** | — |
| Baseline: β/λ proxy only | **0.6048** | — |
| Constant majority class | 0.5024 | — |

P's CI lower bound exceeds 0.5, but it **beats neither simple baseline**. Under the frozen `h2_rule` (CI lower bound > 0.5 **and** better than every baseline in all primary leave-one-out folds), **H2 tends not to hold**.

> `score_gap` is zero in the single-condition smoke test; across the full explore data it has 160 unique values, std 0.0112, and receives a `+6.374` coefficient.

## 6. Tendencies for the pre-registered hypotheses (NOT formal verdicts)

| Hypothesis | Explore signal | Tendency |
|---|---|---|
| H1 (non-monotonic surface + locatable boundary) | all three components have interior extrema; `η*_0` locatable | tends to hold, but effect tiny, most CIs include 0 → weak evidence for a positive-gain window |
| H2 (primary: gain sign predictable a priori) | P 0.5512 < 0.6071 / 0.6048 | **tends not to hold** |
| H3 (opposite derivative signs) | all three curves non-monotonic, no clear opposite monotone derivatives | insufficient / tends not to hold |

Per §4.4 of the design, **the explore stage is for modelling only; formal H1/H2/H3 verdicts must be issued once, in the confirm stage**. The table is a directional readout, not a conclusion.

## 7. Analysis implementation notes

1. Single-factor curves follow the pre-registered definition, including the `tau == 1` constraint; every curve contains the shared `η=0` baseline.
2. The `frozen_criterion.json` digest seals model parameters only and carries no timestamp, so reruns are identical.

## 8. Not completed (belongs to the confirm stage)

- Full-sequence, full-frame confirm: MOT17 21 + DanceTrack 25 + MOT20 4 = **50 sequences / 50387 frames** (explore: 10 sequences / 5000 frames, ~10× smaller)
- Three leave-one-out validations: leave-one-dataset, leave-one-detector (primary); leave-one-noise-family (secondary)
- Official `det.txt` anchor protocol for external validity, reported alongside injection
- Formal H1/H2/H3 verdicts under Holm correction, plus calibration curves
- A5's RQ3 attribution (demoted to exploratory, outside the Holm family)

**Cost estimate**: with frozen `MAIN_RESPONSE = (A4, A0)` only, ≈ 8400 cells, ≈ **5–8 GPU hours** extrapolated from explore throughput.

## 9. Archive scope

The confirm stage was not launched, so this report contains no formal verdicts, no Holm correction and no leave-one-out external validation results. `final_decision` stays `null`; all readouts are stage conclusions and directional tendencies from the explore subset. Had confirm been run, its scale would have been 50 sequences / 50387 frames / ≈8400 cells, using the frozen criterion and strategies for a one-shot validation.

---

## 10. Archive audit

- Confirm stage not launched; the repository contains no full re-run, optimisation, protocol change or added ablation results.
- `final_decision` stays `null`; formal H1/H2/H3 verdicts were not completed.
- Explore output: 5040 cells / 10 shards; frozen config `71dc864e…` and frozen criterion `47c32ece…` are readable and consistent.
- Environment rebuild entry point: `code/tooling/bootstrap_new_instance.sh` (datasets downloaded from a public mirror).
