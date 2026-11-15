# Phase 2 Explore Subset · Auto-Generated Summary (Modelling Only, Not a Formal Verdict)

**English note corresponding to [`EXPLORE_REPORT.md`](EXPLORE_REPORT.md)** — that file is the Chinese
auto-generated output of `code/noise/analyze_explore.py` and is kept as a primary artefact.
The full bilingual narrative is in [`../docs/en/03-phase2-explore.md`](../docs/en/03-phase2-explore.md)
and [`../docs/zh/03-phase2-explore.md`](../docs/zh/03-phase2-explore.md); the complete machine-readable
numbers are in `../results/noise/explore_summary.json`.

- Generated: 2026-11-15T14:04:20+00:00
- Frozen config: `71dc864eb6718fac6c0244aac328b964eb608138ce6e92416f516ebb4be76663` (r4)
- Data: 5040 cells / 840 groups / 10 sequences; all completeness and hash checks passed
- **Per design §4.4 this stage fits criterion P and estimates rough boundary positions only. Formal H1/H2/H3 verdicts are issued once, in the confirm stage — which was never launched.**

## Primary response

| Quantity | Mean | 95% CI |
|---|---|---|
| `ΔAssA` (A4−A0) | +0.001832 | [−0.000044, +0.004326] |
| `ΔIDSW` (lower better) | +31.428571 | [+25.034881, +39.241607] |
| `ΔIDF1` | +0.002923 | [+0.000495, +0.005101] |

Positive `ΔAssA` groups: 49.8%; negative: 50.2%.

## Marginal gain by noise component (mean `ΔAssA` per level)

| `σ_loc` | `ΔAssA` | `ΔIDSW` | `DetA(A0)` | | `β` | `ΔAssA` | `ΔIDSW` | `DetA(A0)` |
|---|---|---|---|---|---|---|---|---|
| 0.000 | +0.005099 | +0.6 | 0.9438 | | 0.000 | +0.005099 | +0.6 | 0.9438 |
| 0.020 | +0.002475 | +6.8 | 0.8266 | | 0.050 | −0.003425 | +0.4 | 0.8979 |
| 0.032 | −0.027189 | +17.6 | 0.7382 | | 0.100 | −0.009714 | +2.9 | 0.8498 |
| 0.050 | +0.013656 | +70.2 | 0.6471 | | 0.200 | −0.001619 | +2.9 | 0.7515 |
| 0.100 | +0.002166 | +130.0 | 0.4773 | | 0.432 | +0.000575 | +10.4 | 0.5279 |
| 0.200 | +0.001690 | +73.1 | 0.2844 | | 0.750 | +0.002757 | +9.8 | 0.2328 |

Boundary estimates (frozen definitions: `η*_0` = first interior extremum of `∂ΔAssA/∂η=0`;
`η_flip` = first η with `ΔIDSW > 0`; `η†` = first η where `|ΔAssA| < |DetA(η) − DetA(0)|`):

| Component | `η*_0` | first sign change | `η_flip` | `η†` | interior extremum |
|---|---|---|---|---|---|
| `σ_loc` | 0.0338 | 0.0210 | 0.0 | 0.02 | yes |
| `β` | 0.1206 | 0.0299 | 0.0 | 0.05 | yes |
| `λ` | 0.0960 | 0.2614 | 0.0 | 0.05 | yes |
| `τ` | — | — | 1.0 | — | no (structurally degenerate) |

## Criterion P (frozen)

Leave-one-sequence-out sign accuracy **0.5512** [0.5190, 0.5857], versus confidence-only baseline
**0.6071**, β/λ baseline **0.6048**, and constant-majority **0.5024**. Under the frozen `h2_rule`,
H2 tends not to hold. Digest `47c32ece…8af1`.

> These are explore-subset readouts, not engineering thresholds: effect magnitudes are < 0.013,
> only 6 of 28 conditions have CIs excluding 0, and no leave-one-out external validation,
> official-det dual protocol or Holm correction was performed.
