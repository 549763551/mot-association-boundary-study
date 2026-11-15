# 02 · Experiment B: Prior-Art Search and Reframing

[Back to README](../../README.md) · [中文](../zh/02-noise-gradient-design.md)

> **Status**: design frozen and executed through the explore stage; confirm stage never launched, `final_decision = null`.

## 1. Why the direction changed

Experiment A's `oracle-detection` protocol removed the detector entirely (`DetA ≈ 0.97`), so even a negative result could not answer the more important question: **under real detector noise, when is fine-grained association actually worth deploying?** The author also wanted a direction that was genuinely unexplored and likely to yield a positive gain.

A systematic prior-art search was run first (2 survey-style searches + 9 targeted searches + 3 full-text/abstract retrievals). It was stricter than expected and **overturned several earlier claims**:

| Presumed gap | Verified result |
|---|---|
| Nobody sweeps a continuous noise axis | **False**. DiffusionTrack (arXiv 2308.09905) sweeps continuous α with `B_det=(1−α)B_det + α·B_noise` and reports MOTA curves |
| "Noise intensity as an independent variable" is open | **Partly occupied**. Robustness Score and Breaking Point are already defined |
| PPLR ported to online MOT is unexplored | **False**. CropTrack already applies k-reciprocal + Jaccard to online frame-wise association; PPLR's Cross Agreement Score is, per the original formula, Jaccard |
| Oracle analysis is new | Already in Tracktor / QDTrack / PigTrack |
| Frame-level gating and abstention are new | Already in When-to-Extract, Deep OC-SORT, AAI, U2MOT, LLTrack, Sentinel |
| Detector vs association: which pays more | Ma et al. (arXiv 2609.22291) already quantify it: detector improvement >10%, refined association <5% |

Two factual corrections: VLA-ReID uses the BEE24 bee dataset, not pedestrian MOT; and "relative residual × explicit modular decomposition × Jaccard × agreement as abstention confidence" leaves only combination-novelty that reviewers would likely call obvious.

**Conclusion**: another gate cannot carry the novelty, and no accuracy gain can be promised.

## 2. "Nobody has done this" narrowed to a falsifiable conjunction

We cannot prove a global negative (not found ≠ does not exist). The defensible claim is: within the searchable literature, **no work was found satisfying all four conditions simultaneously** —

- (a) detection noise swept as **orthogonal, calibrated components**;
- (b) response variable is the **marginal association gain** of fine-grained association over an IoU baseline, not a whole-tracker metric;
- (c) formal definition and measurement of the **sign-flip boundary** of that gain;
- (d) a **label-free, pre-evaluation** criterion predicting the sign of the gain, with leave-one-out external validation.

The closest competitors each miss 2–3 conditions. Novelty therefore lies **in methodology and conclusions, not in mechanism**: turning "is this association worth deploying" from a post-hoc score into a quantity that can be calibrated, mapped and predicted in advance. Cross-agreement abstention, difference fingerprints and modular Jaccard are all demoted to **mechanisms under test / controls**, with no novelty claim.

## 3. Pre-registered design

- **Independent variable** `Θ = (σ_loc, β, λ, τ)`: box jitter (normalised by target height), false-negative rate, false-positive rate, and temporal correlation length of failure states. `Θ̂` is inverted from official `det.txt` so the injection coordinates align with a real detector.
- **Response** `ΔAssA = AssA(s) − AssA(SORT-IoU)`, with `ΔIDSW` and `IDF1`. **HOTA/MOTA are not primary**: noise mainly damages `DetA`, and the geometric mean swamps association differences.
- **Three boundary quantities**: `η*_0` (marginal gain reaches zero / first sign change), `η†` (gain swallowed by detection degradation), `η_flip` (fine association starts manufacturing ID switches).
- **Research questions**: RQ1 shape and boundary of the gain surface; **RQ2 (primary)** predict the sign of `ΔAssA` from detector-output observables alone; RQ3 how much of the association gain reported in the literature actually comes from the detector noise level.
- **Statistics**: cluster bootstrap (1000 resamples, resampling by sequence), 95% CI + Holm correction; explore subset for fitting, confirm stage for one-shot validation; no threshold changes after Phase 0 freeze; both injection and official-det protocols reported, never only the favourable one.
- **Distinction from frame-level gating work**: those ask "use ReID on this frame?"; this asks "deploy it in this scenario at all?".

## 4. Five delegated decisions

The author delegated all decisions ("choose whatever is most advantageous"). Settled as:

| # | Decision | Rationale |
|---|---|---|
| 1 | Main contribution = **boundary + predictability**, not a new gate that gains points | After prior-art search this is the only defensible position, and both signs yield a conclusion |
| 2 | MOT17 + DanceTrack for primary analysis; **MOT20 entirely reserved** for leave-one-out validation in confirm | DanceTrack uses GT injection, bypassing its lack of official det; MOT20 never enters fitting |
| 3 | A5 (set-relative residual) **included but demoted to exploratory**, outside the Holm family | Strengthens RQ3 attribution without inflating multiple comparisons |
| 4 | Two-stage sampling + Phase 0 **SHA-256 sealed freeze** | Per-stage verification prevents silent protocol changes |
| 5 | Write config and code immediately | — |

Two additional tightenings: leave-one-noise-family-out demoted to a secondary readout (small samples, wide CIs, false-negative prone); criterion `P` restricted to ≤8-feature logistic regression so coefficients speak directly to H3.

## 5. Phase 0/1 corrected the design before any GPU run

Inverting `Θ̂` from official `det.txt` produced three facts that changed the plan:

1. **Only TRAIN sequences are calibratable**: MOT17 21 / MOT20 4, not the 42 / 8 in the draft (those included TEST, which has no public GT).
2. **Real noise profile**: median box jitter only **0.020**, median miss rate **0.432**, median consecutive-miss run ≈ **10 frames** → dominant noise is **temporally correlated misses**, not box localisation error. The draft's extreme levels were miscalibrated.
3. Pure quantile anchoring skipped the entire `β ∈ (0, 0.39)` range, exactly where a positive-gain window was most plausible → replaced by `hybrid_v1`: draft low levels + one empirical quantile anchor + a pre-declared extreme upper bound, giving **28 unique conditions**.

**Two engineering problems solved**:

- `geometry_keyed_v1`: only `σ_loc > 0` changes box geometry; under β/λ/τ the surviving TP boxes are bit-identical to GT, so features can be reused → **only 9 of 28 conditions need re-extraction** (down from an estimated ~30 GPU·h).
- `src_gid` side channel: the old `_key(frame, round(box,1))` mapping dropped **63 true-target rows** under injection. With an original-gid channel invisible to the tracker, 17/17 self-tests pass (perfect tracker IDSW = 0; two deliberately injected switches counted as exactly 2).

**Cross-source check**: the original local `MOT17.zip` and the server-side hf-mirror repack produced **field-by-field identical `Θ̂`** for 25 sequences (`CROSS_SOURCE_CALIBRATION_OK`).

Frozen artefact: [`results/noise/frozen_config.json`](../../results/noise/frozen_config.json) r4, `integrity.digest = 71dc864e…4be63`. Every revision is recorded in `revision_history` and all occurred **before any GPU run**.

Full design: [`design/EXPERIMENT_DESIGN_NOISE_GRADIENT.en.md`](../../design/EXPERIMENT_DESIGN_NOISE_GRADIENT.en.md) ([Chinese original](../../design/EXPERIMENT_DESIGN_NOISE_GRADIENT.md)).
