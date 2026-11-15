# 04 · Practical Takeaways and Their Limits

[Back to README](../../README.md) · [中文](../zh/04-practical-takeaways.md)

> All readouts are subject to `final_decision = null`. The items below are grouped by whether they can be written into code or change an engineering decision, and the limits of each are stated explicitly.

## Tier 1: directly usable implementation rules (save cost / prevent errors)

| Finding | How to use it | Where it does not apply |
|---|---|---|
| **Real detector noise is dominated by temporally correlated misses**: median box jitter 0.020 (2% of target height), median miss rate 0.432, median consecutive-miss run ≈ 10 frames | Basis for resource allocation: **temporal recovery on the detector side / track keep-alive under occlusion / re-detection recall**, not tighter box alignment or finer appearance scoring. For a 100 px tall pedestrian, median localisation error is ~2 px — little is left to squeeze geometrically | This is the profile of MOT17/MOT20 **official det**, not of all detectors; modern YOLOX-based trackers may differ; only 25 TRAIN sequences were measured |
| **MOT20 `det.txt` column 7 is a 0/1 validity flag, not a confidence**; only MOT17's 21 det files have continuous scores | Bake it into the data loader: route by column semantics, never feed MOT20's 0/1 into a gate or ranking as a score | A parsing rule only; does not change model accuracy |
| **`geometry_keyed_v1`**: only `σ_loc` changes box geometry; under β/λ/τ the surviving TP boxes are bit-identical to GT | When running noise-robustness evaluations or ablations, key the feature cache on "did geometry change". Here only **9 of 28** conditions needed re-extraction | Saves **experiment compute**, not online inference cost |
| **Three evaluation rules**: official HOTA matches with `global_alignment×IoU`, so the TP set depends on tracker id assignment; `A1` discards low-score boxes, making `DetA` incomparable; after perturbation, identity should be mapped through a `src_gid` side channel rather than box-coordinate keys | In tracker-comparison scripts: if detection sets differ, do not compare HOTA/DetA directly — report `AssA` and state the caveat; always use the original-gid side channel for injection experiments. The bundled lightweight AssA/DetA implementation matches official `eval_sequence` with **diff = 0**, so a full TrackEval directory tree is unnecessary | Concerns evaluation correctness only; does not change the tracker |

These four form a reusable set of implementation rules: lower evaluation cost, avoid protocol errors, and provide a basis for resource allocation.

## Tier 2: changes decisions, cannot be deployed

- **"Ambiguity is not confidence"**: criterion `P` scores 0.5512 leave-one-sequence-out, below the simple baselines 0.6071 / 0.6048. Its use is **negative** — top-1/top-2 margins should not be used as a ReID gate or confidence. It offers no replacement, comes from an explore subset, and was never formally adjudicated, so it is weak evidence only.
- Together with Ma et al.'s quantitative result (detector improvement >10% vs refined association <5%), the noise profile above supports a **priority ordering: fix the detector first, then discuss association**. That is a planning input, not an algorithm.

## Tier 3: explicitly not usable as engineering thresholds

The boundary quantities `η*_0` = 0.034 / 0.121 / 0.096, `η†` ≈ 0.02–0.05 and `η_flip = 0` **must not be used to set thresholds**:

- all magnitudes < 0.013, only 6 of 28 conditions have CIs excluding 0;
- only 5 + 5 sequences × 500 frames were run;
- no MOT20 leave-one-out external validation, no injection / official-det dual protocol, no Holm correction.

They are **descriptive readouts**, not calibrated results. Configuring a system from them would mean treating the explore stage as the confirm stage.

## A separate class of reusable assets (unrelated to research conclusions)

Environment rebuild scripts (datasets downloaded directly from a public mirror), per-sequence-sharded result synchronisation, and an OS-lease-lock backup daemon. These are engineering assets — see [`code/tooling/`](../../code/tooling/) and [Reproduction & archive](05-reproduction-and-archive.md).

## Summary

The conclusions in this archive concern resource allocation, compute savings, evaluation correctness and negative guidelines. They do not include an algorithm that can be deployed to raise tracking accuracy.
