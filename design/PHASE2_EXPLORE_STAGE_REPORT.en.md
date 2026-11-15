# Phase 2 Explore Subset · Stage Report (Halted for Review)

**English counterpart of [`PHASE2_EXPLORE_STAGE_REPORT.md`](PHASE2_EXPLORE_STAGE_REPORT.md)**
Generated 2026-11-15 · **Status: explore complete, criterion frozen; confirm stage NOT launched; archived after author review**

- Frozen config: `results/noise/frozen_config.json` r4, sha256 `71dc864e…4be63` (unchanged throughout)
- Frozen criterion: `results/noise/frozen_criterion.json`, sha256 `47c32ece…8af1` (reproducible, rerun digest identical)

---

## 1. One-sentence conclusion

Across 5040 controlled units, **the marginal association gain of fine-grained association (cross-agreement abstention `A4`) over the IoU baseline `A0` is near zero and statistically insignificant, while ID switches increase significantly; the label-free criterion `P` designed to predict the sign of that gain before evaluation does not beat two simple baselines**. Per the pre-registered design this points toward a corrective negative conclusion; but **the formal verdict was never completed**, so this must not be written as "falsified".

## 2. Execution and completeness

| Item | Result |
|---|---|
| Explore scale | 10 sequences (MOT17 5 + DanceTrack 5) × first 500 frames × 28 frozen conditions × 3 seeds × 6 strategies |
| Output | **5040 cells / 840 groups**, 5071 s (≈ 84.5 min) |
| Completeness gate | **11/11 passed**: row count, single hash, stage, seeds, strategies, conditions, benchmarks, sequence count, group count, 6 strategies per group, 8 observables |
| Metric fidelity | Our `AssA`/`DetA` vs TrackEval official `HOTA.eval_sequence`: **difference 0** (17/17 self-tests locally and on the server) |
| Bandwidth | Datasets on the server's persistent volume; only script/report-sized files transferred |

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

Two facts must be stated plainly:

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

> Correction to an earlier statement: `score_gap` is zero only in the single-condition smoke test. Across the full explore data it has 160 unique values, std 0.0112, and receives a `+6.374` coefficient — it is **not** a zero-variance degenerate feature.

## 6. Tendencies for the pre-registered hypotheses (NOT formal verdicts)

| Hypothesis | Explore signal | Tendency |
|---|---|---|
| H1 (non-monotonic surface + locatable boundary) | all three components have interior extrema; `η*_0` locatable | tends to hold, but effect tiny, most CIs include 0 → weak evidence for a positive-gain window |
| H2 (primary: gain sign predictable a priori) | P 0.5512 < 0.6071 / 0.6048 | **tends not to hold** |
| H3 (opposite derivative signs) | all three curves non-monotonic, no clear opposite monotone derivatives | insufficient / tends not to hold |

Per §4.4 of the design, **the explore stage is for modelling only; formal H1/H2/H3 verdicts must be issued once, in the confirm stage**. The table is a directional readout, not a conclusion.

## 7. Two analysis-script fixes (both before any formal verdict)

1. The first `families()` omitted the `tau == 1` constraint, mixing two `β×τ` interaction conditions into the β single-factor curve and producing duplicate η values plus a spurious interior extremum. Fixed to the pre-registered definition; the shared `η=0` baseline was added to every curve.
2. The first `frozen_criterion.json` digest included `fit_utc`, so reruns changed the hash. It now seals model parameters only; `RERUN_DIGEST` verified identical.

Both affect **analysis scripts only** — no experimental data, injector, frozen config or computed cell was altered.

An additional operational fix: the local backup daemon deadlocked for 52 minutes because `os.remove` on its lock file was blocked by a safe-delete policy. It was replaced by an OS lease lock and restarted; all 5040 cells were caught up, snapshots `20261115_212608` / `…212701`.

## 8. Not completed (belongs to the confirm stage)

- Full-sequence, full-frame confirm: MOT17 21 + DanceTrack 25 + MOT20 4 = **50 sequences / 50387 frames** (explore: 10 sequences / 5000 frames, ~10× smaller)
- Three leave-one-out validations: leave-one-dataset, leave-one-detector (primary); leave-one-noise-family (secondary)
- Official `det.txt` anchor protocol for external validity, **reported alongside** injection, never only the favourable one
- Formal H1/H2/H3 verdicts under Holm correction, plus calibration curves
- A5's RQ3 attribution (demoted to exploratory, outside the Holm family)

**Cost estimate**: with frozen `MAIN_RESPONSE = (A4, A0)` only, ≈ 8400 cells, ≈ **5–8 GPU hours** extrapolated from explore throughput; local bandwidth still near zero, but the server must stay online and carries recycling risk.

## 9. Recommendation: close here, do not auto-launch confirm

1. The primary contribution ("pre-evaluation predictability") already fails to beat simple baselines within explore; a full confirm would most likely just formalise the negative conclusion.
2. `ΔAssA` is insignificant on average and `ΔIDSW` significantly worse, with 5040 controlled units supporting the direction "under oracle/injected detection there is almost no positive-gain window worth deploying".
3. Per the author's rule — when there is nothing worth discovering, do not start new experiments, do not change protocols to manufacture a positive result, do not keep burning server and bandwidth for output; instead compile a complete report for review.

**Mandatory limitation**: `final_decision` remains `null`. Confirm never ran, so strictly this is an "explore-stage conclusion plus tendency", **not** "H1/H2/H3 falsified". It is not repackaged as "a new direction found" either. Experiment A also remains terminated.

## 10. Archive status (2026-11-15 22:16, author confirmed closure)

- **Confirm stage not launched.** No instance re-request, no full re-run, no optimisation, no new artefact, no protocol change, no added ablation.
- `final_decision` stays `null`. This report is an "explore-stage conclusion plus directional tendency", **not** a formal H1/H2/H3 verdict, **not** "falsified", and not packaged as "a new direction".
- Final backup audit: server `NO_EXPERIMENT_PROCESS`; local mirror 41 files, `cells_explore` **10 shards / 5040 cells** complete; `run_explore.json` frozen hash `71dc864e…` and `frozen_criterion.json` `47c32ece…` both readable and consistent; **79** timestamped snapshots accumulated.
- Final backup pass at 22:16:38 returned `no change` (fully synchronised); the backup daemon was then stopped.
- Recovery entry points: data via `bootstrap_new_instance.sh` on any new instance (15 GB downloaded server-side, zero local bandwidth); backup daemon via `bash backup_watch.sh 20`.
