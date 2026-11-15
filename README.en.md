# Where Fine-Grained Association Stops Paying Off: An Archive of Two Experiments

**[中文版](README.md)** · Archived 2026-11-15 · Status: **static archive; experiments terminated**

> **Status**
> This repository archives two real-benchmark experiments. Experiment A was terminated before full evaluation completed; Experiment B was archived after its pre-registered *explore* stage, with the *confirm* stage never launched.
> `final_decision` is `null` for **both** experiments: the pre-registered formal verdict procedures were not completed.
> The repository reports completed measurements, stage-level conclusions and reproduction material only.

---

## The question

A recurring implicit assumption in the MOT literature is that when detector output is ambiguous, adding more sophisticated appearance association (ReID, relative-difference scoring, agreement-based gating) yields a meaningful gain. Two controlled experiments measure the **boundary of that assumption**. The result:

**Under both `oracle-detection / association-only` and controlled injected detection noise, the marginal association gain of fine-grained association over an IoU baseline is near zero and statistically insignificant, while ID switches increase significantly. Moreover, the primary research question — predicting the *sign* of that gain before evaluation — does not beat simple baselines.**

Pre-registration, a frozen grid and hash sealing make the results reproducible and independently checkable.

## The two experiments

| | Experiment A · Confusion-gated pairwise-difference arbitration | Experiment B · Detection-noise gradient / association-gain boundary |
|---|---|---|
| Origin | Synthetic study: sparse difference signals let relative discrimination cross the separability threshold ~`√D` earlier than global discrimination | After a systematic prior-art search, the study targets methodological boundaries and predictability |
| Protocol | DanceTrack / MOT17 / MOT20, unified `oracle-detection`, GT boxes filtered per TrackEval | Orthogonally calibrated `Θ=(σ_loc, β, λ, τ)`; response `ΔAssA = AssA(s) − AssA(SORT-IoU)` |
| Strategies | `S0_motion`, `S1_global`, `S2_countgated`, six `S3_*` variants | Six `A0`–`A5` associators, incl. cross-agreement abstention `A4` |
| Scale | Target 36 sequences; completed DanceTrack 25/25 and MOT17 7/7 feature extraction | Explore: 5040 cells / 840 groups / 84.5 min, 10 sequences × 500 frames |
| Archive status | Terminated after feature extraction; full evaluation and formal verdict not completed | Explore complete, criterion frozen; confirm stage not launched |
| Verdict | `final_decision = null` | `final_decision = null` |

## Key numbers

**Experiment B, explore stage (cluster bootstrap, 1000 resamples, resampling by sequence)**

| Quantity | Mean | 95% CI | Reading |
|---|---|---|---|
| `ΔAssA` (A4−A0) | **+0.001832** | **[−0.000044, +0.004326]** | includes 0 → no significant average gain |
| `ΔIDSW` (lower is better) | **+31.43** | **[+25.03, +39.24]** | significantly positive → significantly worse |
| `ΔIDF1` | +0.002923 | [+0.000495, +0.005101] | significant but negligible magnitude |

- `ΔAssA > 0` in **49.76%** of groups (≈ coin flip). Only 6 of 28 conditions have CIs excluding 0, all with magnitude < 0.013.
- `η_flip = 0`: at zero noise `ΔIDSW` is already +0.6 — fine-grained association starts manufacturing ID switches from the clean regime.
- Criterion `P` (≤8 label-free features, L2 logistic regression, [frozen](results/noise/frozen_criterion.json), sha256 `47c32ece…8af1`): leave-one-sequence-out sign accuracy **0.5512** [0.5190, 0.5857], **below** the confidence-only baseline 0.6071 and the β/λ baseline 0.6048 → by the frozen `h2_rule`, **H2 tends not to hold**.

**Evaluation metric verified**: the `AssA` / `DetA` implementation matches TrackEval's official `HOTA.eval_sequence` with **difference 0** (17/17 self-tests locally and 17/17 on the server).

## What is actually usable

See [Practical takeaways](docs/en/04-practical-takeaways.md) · [可落地结论（中）](docs/zh/04-practical-takeaways.md). In brief:

- **Where to invest**: inverting `Θ̂` from 25 official `det.txt` files shows real detectors have a median box jitter of only **0.020** (2% of target height), but a median miss rate of **0.432** and median consecutive-miss run length of **~10 frames**. The dominant noise is **temporally correlated misses**, not box localisation error. Resources are better spent on temporal recovery and track keep-alive under occlusion than on tighter box alignment or finer appearance scoring.
- **Save compute**: `geometry_keyed_v1` — only `σ_loc` changes box geometry; under β/λ/τ the surviving TP boxes are bit-identical to GT, so only **9 of 28** conditions need feature re-extraction.
- **Evaluation implementation rules**: official HOTA matches with `global_alignment×IoU`, so the TP set depends on tracker id assignment; `A1` discards low-score boxes, making `DetA` incomparable; after perturbation, identity should be mapped through a `src_gid` side channel rather than box-coordinate keys.
- **Negative guideline**: "ambiguity equals confidence" does not hold here; top-1/top-2 margins should not be used as a ReID gate. No replacement is offered.
- **Not engineering thresholds**: `η*_0` / `η†` / `η_flip` are descriptive readouts from an explore subset — small effects, wide CIs, no leave-one-out external validation, no Holm correction.

## Documentation

| 中文 | English |
|---|---|
| [01 实验 A：设计与中止状态](docs/zh/01-original-experiment.md) | [01 Experiment A](docs/en/01-original-experiment.md) |
| [02 实验 B：查新与设计](docs/zh/02-noise-gradient-design.md) | [02 Experiment B: prior art & design](docs/en/02-noise-gradient-design.md) |
| [03 实验 B：探索阶段结果](docs/zh/03-phase2-explore.md) | [03 Experiment B: explore-stage results](docs/en/03-phase2-explore.md) |
| [04 可落地结论与边界](docs/zh/04-practical-takeaways.md) | [04 Practical takeaways](docs/en/04-practical-takeaways.md) |
| [05 复现、完整性与归档](docs/zh/05-reproduction-and-archive.md) | [05 Reproduction & archive](docs/en/05-reproduction-and-archive.md) |

Primary material (Chinese originals + English counterparts): [`design/`](design/)

## Layout

```
README.md / README.en.md        bilingual entry points
docs/zh/ · docs/en/             5 paired documents each
design/                         pre-registered design, stage report (bilingual), analysis note
results/original/               exp. A: dry-run report, difference report, state record, figure
results/noise/                  exp. B: frozen config, frozen criterion, summary, calibration,
                                5040 cells (10 per-sequence JSONL shards)
code/original/ · code/noise/    complete runnable code for both experiments
code/tooling/                   deployment, resume and incremental-backup scripts
background/                     synthetic E1-E4 (motivation only, NOT a third experiment)
SHA256SUMS.txt                  sha256 of every published file
```

**Not included**: the 15 GB datasets, OSNet weights, extracted `.npy` features, private keys or server credentials. Datasets are downloaded directly from a public mirror — see [Reproduction](docs/en/05-reproduction-and-archive.md).

## Integrity

- Frozen config [`results/noise/frozen_config.json`](results/noise/frozen_config.json) r4, `integrity.digest = 71dc864e…4be63`, matching the per-row `frozen_sha256` embedded in all 5040 cells. **Never modified.**
- Frozen criterion [`results/noise/frozen_criterion.json`](results/noise/frozen_criterion.json) digest is reproducible by rerunning the analysis.
- Explore completeness gate **11/11 passed** (row count, single hash, stage, seeds, strategies, conditions, benchmarks, sequence count, group count, 6 strategies per group, 8 observables).
- Cross-source check: the original MOT17 archive and a public-mirror repack produced **field-by-field identical `Θ̂`** across 25 sequences.
- Per-file sha256 in [`SHA256SUMS.txt`](SHA256SUMS.txt).

## Not included in this archive

The confirm stage (50 sequences / 50387 frames / ~8400 cells / estimated 5–8 GPU hours) was never launched. Consequently the archive contains no formal H1/H2/H3 verdicts, no Holm correction, no three leave-one-out validations, no official-`det.txt` anchor protocol reported alongside injection, and no A5 RQ3 attribution.

## Licence and citation

Code is MIT; documents and experimental data are CC BY 4.0 (see [LICENSE](LICENSE)). If citing the negative result or the `Θ̂` calibration, please state that this is an **explore / terminated-stage archive**, not a formal verdict.
