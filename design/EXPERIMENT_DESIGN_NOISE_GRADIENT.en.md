# Detection-Noise Gradient Protocol: Mapping and Predicting the Marginal Gain of Fine-Grained Association

**Pre-registered experiment design · 2026-11-15 · English counterpart of [`EXPERIMENT_DESIGN_NOISE_GRADIENT.md`](EXPERIMENT_DESIGN_NOISE_GRADIENT.md)**

> This is the English counterpart of the Chinese primary design document. It carries the same
> pre-registration content (research questions, hypotheses, variables, metrics, statistics,
> frozen grid, decisions and stopping rule). The Chinese original additionally contains the full
> reference list with retrieved URLs and the verbatim revision log.
>
> **Status at publication**: Phase 0/1 complete, explore stage complete; **confirm stage not launched**
> by the author's decision. `final_decision = null`. This is an archived design, not a completed study.

---

## 1. Position relative to the terminated experiment

This is a **new experiment**, not a continuation or optimisation of the "confusion-gated pairwise-difference arbitration" study. That study was halted on 2026-11-15 with `final_decision = null` (full evaluation never completed; not falsified). It used `oracle-detection / association-only` with `DetA ≈ 0.97`, which isolates association but cannot answer when fine-grained association is worth deploying under real detector noise. The new experiment addresses exactly that gap.

## 2. Systematic prior-art search

Two survey-style searches, nine targeted searches and three full-text/abstract retrievals were performed. Findings that **overturned earlier assumptions**:

| Presumed gap | Verified |
|---|---|
| Continuous noise sweep is unexplored | DiffusionTrack (arXiv 2308.09905): `B_det=(1−α)B_det + α·B_noise`, continuous α, MOTA curves |
| Noise intensity as an independent variable is open | Robustness Score and Breaking Point already defined (ict.od.ua; abstract verified) |
| PPLR ported to online MOT is unexplored | CropTrack applies k-reciprocal + Jaccard to online frame-wise association; PPLR's Cross Agreement Score is Jaccard |
| Oracle analysis is new | Tracktor / QDTrack / PigTrack |
| Frame-level gating and abstention are new | When-to-Extract, Deep OC-SORT, AAI, U2MOT, LLTrack, Sentinel |
| Detector vs association attribution is new | Ma et al. (arXiv 2609.22291): detector improvement >10%, refined association <5% |

Corrections: VLA-ReID uses the BEE24 bee dataset, not pedestrian MOT; "relative residual × explicit modular decomposition × Jaccard × agreement as abstention confidence" leaves only combination novelty likely to be judged obvious.

**Implication**: another gate cannot carry the contribution, and no accuracy gain can be promised.

## 3. The novelty claim, narrowed to a falsifiable conjunction

We cannot prove a global negative (not found ≠ does not exist). The defensible statement is: within the searchable literature, **no work was found satisfying all four conditions simultaneously**:

- (a) detection noise swept as **orthogonal, calibrated components**;
- (b) response = **marginal association gain** of fine-grained association over an IoU baseline, not a whole-tracker metric;
- (c) formal definition and measurement of the **sign-flip boundary** of that gain;
- (d) a **label-free, pre-evaluation** criterion predicting the sign of the gain, with leave-one-out external validation.

Closest competitors each miss 2–3 conditions. Novelty lies in **methodology and conclusions**: turning "is this association worth deploying" from a post-hoc score into a calibratable, mappable, predictable quantity. Cross-agreement abstention, difference fingerprints and modular Jaccard are demoted to **mechanisms under test / controls** and carry no novelty claim.

Search limitations and a downgrade path are recorded in the Chinese original.

## 4. Research questions and hypotheses

- **RQ1** — What is the shape of the marginal-gain surface over detection noise, and where are its boundaries?
- **RQ2 (primary)** — Can the **sign** of `ΔAssA` be predicted before evaluation, using only label-free observables of detector output?
- **RQ3** — How much of the association gain reported in the literature is actually attributable to the detector noise level?

| ID | Hypothesis | Pre-registered decision rule |
|---|---|---|
| H1 | The gain surface is non-monotonic with locatable boundaries `η*_0`, `η†`, `η_flip` | Bootstrap CI of boundary estimates; interior extrema required |
| H2 | Gain sign is predictable a priori by criterion `P` | Frozen `h2_rule`: CI lower bound > 0.5 **and** better than every baseline in all primary leave-one-out folds |
| H3 | Partial derivatives w.r.t. noise components have opposite signs | Sign pattern across single-factor curves |

H2 is the primary contribution. **If H2 fails, the result is still publishable as a corrective negative finding**: the widely assumed "ambiguity equals confidence" premise would be shown not to hold under this protocol. This is why both signs yield a conclusion.

## 5. Variables, mechanisms and metrics

**Independent variable** `Θ = (σ_loc, β, λ, τ)`

| Component | Meaning | Calibration |
|---|---|---|
| `σ_loc` | box jitter, normalised by target height | inverted from official `det.txt` |
| `β` | false-negative (miss) rate | ditto |
| `λ` | false-positive rate | ditto |
| `τ` | temporal correlation length of failure states | ditto |

Using official `det.txt` to invert `Θ̂` aligns injection coordinates with a real detector; DanceTrack has no official det and is handled by GT-based injection, so this is not an obstacle.

**Strategies (all other factors fixed)**: `A0` SORT-IoU baseline; `A1` ByteTrack-style two-stage; `A2` global ReID; `A3` cross-agreement abstention; `A4` the primary abstention variant; `A5` set-relative residual (exploratory, outside the Holm family).

**Response**: `ΔAssA = AssA(s) − AssA(SORT-IoU)`, with `ΔIDSW` and `IDF1`. **HOTA/MOTA are not primary** — noise mainly damages `DetA`, and the geometric mean would swamp association differences. `AssA` follows the official definition: mean of AssA over 19 IoU thresholds, with HOTA's Hungarian matching using `global_alignment_score × IoU`.

**Boundary quantities**:
- `η*_0` — first interior extremum / first sign change of `ΔAssA`;
- `η†` — smallest η where `|ΔAssA| < |DetA(η) − DetA(0)|`;
- `η_flip` — smallest η where fine association starts manufacturing ID switches (`ΔIDSW > 0`).

**Criterion `P`**: L2 logistic regression on ≤8 label-free observables (`mean_score`, `p10_score`, `score_gap`, `mean_pair_iou`, `det_per_frame`, `scale_cv`, `common_mode_energy`, `low_score_frac`), target `sign(ΔAssA)`. The ≤8-feature cap is deliberate: coefficients must remain interpretable for H3.

**Distinction from frame-level gating**: prior work asks "use ReID on this frame?"; this asks "deploy it in this scenario at all?".

## 6. Frozen grid and Phase 0/1 corrections

Grid rule `hybrid_v1`, sealed at r4: `integrity.digest = 71dc864e…4be63`, **28 unique conditions**. All revisions are in `revision_history` and occurred before any GPU run.

Phase 1 (calibration, no GPU experiment) corrected three design assumptions:

1. Only TRAIN sequences are calibratable: **MOT17 21 / MOT20 4**, not 42 / 8 (the latter include TEST without public GT).
2. Real detector `Θ̂`: median box jitter **0.020**, median miss rate **0.432**, median consecutive-miss run ≈ **10 frames** → dominant noise is **temporally correlated misses**, not localisation error. The draft's extreme levels were miscalibrated.
3. Pure quantile anchoring skipped `β ∈ (0, 0.39)` — the most plausible positive-gain window. `hybrid_v1` restores the draft's low levels, keeps one empirical quantile anchor and a pre-declared extreme upper bound.

Two engineering results, both sealed into the frozen config and self-test:

- `geometry_keyed_v1`: only `σ_loc > 0` changes box geometry, so **9 of 28** conditions require feature re-extraction (estimated ~30 GPU·h avoided).
- `src_gid` side channel: the old box-coordinate identity mapping dropped **63 true-target rows** under injection. 17/17 self-tests pass, including perfect-tracker IDSW = 0 and exactly 2 switches for 2 injected swaps.

**Cross-source check**: original local `MOT17.zip` vs server-side hf-mirror repack — `Θ̂` field-by-field identical over 25 sequences (`CROSS_SOURCE_CALIBRATION_OK`).

## 7. Statistics and stopping rule

- Two-stage sampling: **explore subset fits, confirm stage validates once**. Explore is for modelling only; formal H1/H2/H3 verdicts are issued once in confirm.
- Phase 0 SHA-256 sealing: no threshold changes, no added/removed conditions, no protocol changes after the freeze. Every stage verifies the hash.
- Cluster bootstrap, 1000 resamples, resampling by sequence; 95% CI; Holm correction for the primary family.
- Both injection and official-det protocols reported; **never only the favourable one**.
- Primary response frozen as `MAIN_RESPONSE = (A4, A0)`.
- Leave-one-noise-family-out is demoted to a secondary readout (small samples, wide CIs, false-negative prone).
- **Stopping rule**: after the explore stage, present the frozen grid and results to the author. Any further GPU work requires explicit approval. If results contain no worthwhile finding, close the project rather than starting new experiments, changing protocols or adding ablations to manufacture a positive result.

## 8. Delegated decisions

The author delegated all five decisions ("choose whatever is most advantageous") on 2026-11-15:

| # | Decision |
|---|---|
| 1 | Main contribution = **boundary + predictability**, not a new gate that gains points |
| 2 | MOT17 + DanceTrack primary; **MOT20 entirely reserved** for leave-one-out external validation |
| 3 | A5 included but demoted to exploratory, outside the Holm family |
| 4 | Two-stage sampling + Phase 0 SHA-256 sealed freeze |
| 5 | Write the full pre-registered config and code immediately |

## 9. Execution status and final decision

- Phase 0/1 complete (calibration, freeze, 17/17 self-tests locally and on server, cross-source check).
- Phase 2 **explore** complete: 5040 cells / 840 groups / 5071 s, completeness gate 11/11, frozen hash unchanged. See [`PHASE2_EXPLORE_STAGE_REPORT.en.md`](PHASE2_EXPLORE_STAGE_REPORT.en.md) and `../results/noise/explore_summary.json`.
- Criterion `P` frozen (`47c32ece…8af1`), reproducible by rerunning.
- **Confirm stage not launched.** On 2026-11-15 22:16 the author reviewed the explore report and chose to close the project: no full re-run, no optimisation, no new artefact, no protocol change, no added ablation.
- `final_decision` remains `null`. Explore results are a **stage conclusion plus directional tendency**: `ΔAssA` not significant, `ΔIDSW` significantly worse, criterion `P` below simple baselines. **Not a falsification, not a new direction.**

Had confirm been run, its scale would have been 50 sequences / 50387 frames / ≈8400 cells / estimated 5–8 GPU hours, using only frozen strategies and a frozen criterion, with no re-tuning.

## 10. Known engineering pitfalls (recorded in the design)

- Changing boxes invalidates cached features; `geometry_keyed_v1` limits re-extraction.
- Det boxes do not coincide with GT boxes → IDSW identity back-mapping must be redone and `idsw_consistency` re-verified (this caught the 63-row loss).
- `DetA` degradation dominates HOTA → keep AssA/IDSW as primary.
- Large condition count → strict separation of primary/exploratory analyses, Holm correction, no result-driven condition changes.
- Result-driven protocol switching → η must be frozen at Phase 0 and both protocols reported.
- Per-file backup size cap → results sharded per sequence as JSONL.

## 11. Scope limitations

- `oracle`/injection protocols, not a deployed detector in the loop; external validity via the official-det anchor was pre-registered but never executed.
- Explore used 10 sequences × 500 frames; MOT20 was never used for fitting.
- Search-based novelty claims are bounded by what was retrievable; "not found" is not "does not exist".
- No formal H1/H2/H3 verdict exists; all hypothesis statements are tendencies.

For the complete reference list with retrieved URLs, the verbatim revision log, and the full Chinese text, see [`EXPERIMENT_DESIGN_NOISE_GRADIENT.md`](EXPERIMENT_DESIGN_NOISE_GRADIENT.md).
