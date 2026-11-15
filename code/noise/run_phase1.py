#!/usr/bin/env python3
"""Phase 0 freeze + Phase 1 calibration / self-test orchestrator.

Run on the LOCAL machine first (--source local): calibration reads the 15 GB
zips already on disk, so it costs ZERO network traffic. Then re-run on the
SERVER (--source server) to verify the frozen config reproduces from the
unpacked data. The frozen file is hash-sealed: any later edit is detected.

    python run_phase1.py --source local  --out .
    python run_phase1.py --source server --out . --verify-only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import det_calibration as dc
import phase1_selftest as st
from noise_injection import grid_conditions

PROTOCOL = "noise_gradient_v1"
FROZEN_FILE = "frozen_config.json"


def _hash(obj: dict) -> str:
    body = {k: v for k, v in obj.items() if k != "integrity"}
    s = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def build_frozen(calib: dict) -> dict:
    grid = calib["grid"]
    conds = [t.as_dict() for t in grid_conditions(grid)]
    cfg = dict(
        protocol=PROTOCOL,
        frozen_at=_utc(),
        freeze_revision=4,
        revision_history=[
            "r1 quantile_anchor_v1: first freeze, selftest crashed on missing "
            "import math (no experiment had been run)",
            "r2 quantile_anchor_v1 + fixed upper bound: selftest 15/15 passed, "
            "but grid skipped the entire low-noise beta region",
            "r3 hybrid_v1: restore design-v1 low levels, keep one Theta-hat "
            "quantile anchor and the pre-declared upper bound",
            "r4 (FINAL): add the geometry-keyed feature-cache rule after a "
            "GPU-cost estimate showed naive full re-extraction needs ~30 h; "
            "all revisions made BEFORE any GPU run or DeltaAssA observation",
        ],
        frozen_by="user delegated all 5 decisions ('按照你认为有利的就行'), 2026-11-15",
        status="FROZEN - do not edit; hash-sealed",

        decisions=dict(
            D1_primary_contribution="boundary measurement + pre-evaluation "
                                    "label-free gain-sign predictability (NOT a new gate)",
            D1_rationale="novelty search showed every mechanism is already "
                         "covered; only the methodological combination survives",
            D2_datasets=dict(main=["mot17", "dancetrack"],
                             external_validation=["mot20"],
                             dancetrack_note="GT-based injection; no official det needed"),
            D3_A5="INCLUDED but exploratory: MOT17 explore-subset only, used "
                  "solely for RQ3 attribution, excluded from H1/H2/H3 and from "
                  "the Holm family",
            D4_two_stage_and_freeze=True,
            D4_note="Phase 0 hash-seal; explore fits P and locates boundaries; "
                    "confirm is one-shot with no threshold/condition changes",
            D5_execute_now=True,
        ),

        research_questions=dict(
            RQ1="response surface of marginal association gain vs orthogonal "
                "detection-noise components; is there a sign-flip boundary?",
            RQ2="MAIN: predict sign/magnitude of the gain BEFORE evaluation, "
                "without GT labels, and transfer across datasets/detectors",
            RQ3="where do published association gains sit on that surface; how "
                "much is attributable to detector noise rather than mechanism",
        ),
        hypotheses=dict(
            H1="DeltaAssA non-monotone in >=1 component with finite bootstrap CI "
               "for eta*_0",
            H2="label-free criterion predicts DeltaAssA sign significantly "
               ">0.5 and beats simple baselines under leave-one-out validation",
            H3="component partial derivatives have opposite signs with "
               "non-overlapping CIs",
        ),

        theta=dict(components=["sigma_loc", "beta", "lambd", "tau"],
                   sigma_loc="centre/scale jitter normalized by target height",
                   beta="per-target frame deletion rate (FN)",
                   lambd="false positives per GT box per frame",
                   tau="temporal correlation length L in frames (control variable, "
                       "NOT claimed novel)"),
        grid=grid,
        conditions=conds,
        n_conditions_unique=len(conds),
        n_conditions_note="design v1 quoted 36; the de-duplicated enumeration is "
                          "smaller because the all-zero baseline recurs across "
                          "single-factor sweeps. Difference is an accounting "
                          "correction made BEFORE any result, not a result-driven cut.",

        response=dict(
            delta_assa="AssA(strategy, Theta) - AssA(A0_IoU, Theta)",
            delta_idsw="IDSW(strategy, Theta) - IDSW(A0_IoU, Theta)  (negative is better)",
            primary_metrics=["AssA", "IDSW", "IDF1"],
            forbidden_as_primary=["HOTA", "MOTA"],
            reason="noise degrades DetA; HOTA=sqrt(DetA*AssA) masks association gain",
            secondary=["DetA", "Frag", "MOTA", "HOTA", "reid_calls", "trigger_rate"],
            boundaries=dict(eta_star_0="d DeltaAssA / d eta = 0",
                            eta_dagger="|DeltaAssA(eta)| < |DeltaDetA(eta)|",
                            eta_flip="minimum eta where DeltaIDSW > 0"),
        ),

        strategies=dict(
            A0="SORT / IoU-only (reference for every Delta)",
            A1="ByteTrack-style two-stage low-score recovery",
            A2="OSNet appearance fusion + Hungarian",
            A3="original PDE arbitration / confusion trigger (negative control)",
            A4="cross-agreement abstention (Jaccard gate) - MEASURED mechanism, "
               "novelty NOT claimed (PPLR / CropTrack / AAI / LLTrack precede it)",
            A5="set-relative residual rescoring (VLA-ReID-like CAS approximation) - "
               "EXPLORATORY, RQ3 only",
            fixed="identical cached OSNet FP32 features for all strategies",
        ),

        sampling=dict(
            explore="5 sequences x first 500 frames per benchmark",
            confirm="all sequences, all frames, one shot",
            seeds=[11, 23, 37],
            min_seeds_per_cell=3,
            calibration_of_grid="official TRAIN det.txt Theta-hat quantiles, "
                                "computed once in Phase 1 and then frozen",
        ),
        feature_cache=dict(
            rule="geometry_keyed_v1",
            reextract_rule="only sigma_loc>0 invalidates cached OSNet features",
            beta_tau="surviving TP boxes are bit-identical to GT boxes, so the "
                     "GT-keyed feature cache is reused unchanged",
            lambda_note="TP features reused; only injected FP crops need "
                        "extraction (~lambda x n_GT extra crops)",
            conditions_requiring_reextraction=sum(
                1 for t in conds if t["sigma_loc"] > 0),
            conditions_total=len(conds),
            cost_note="naive full re-extraction over all conditions was "
                      "estimated at ~30 GPU-hours; geometry keying removes most "
                      "of it. Structural fact, not a result-driven approximation.",
            verification="phase1_selftest checks "
                         "beta_lambda_tau_reuse_gt_features and "
                         "sigma_loc_forces_reextraction",
        ),

        statistics=dict(
            bootstrap_n=1000, bootstrap_seed=20261115,
            ci=0.95, correction="Holm", alpha=0.05,
            primary_family=["H1", "H2", "H3"],
            leave_one_out_primary=["dataset", "detector"],
            leave_one_out_secondary=["noise_component_family"],
            predictor="logistic regression, <=8 features, coefficients reported",
            predictor_baselines=["detection confidence only", "beta/lambda only",
                                 "constant majority class"],
            h2_rule="H2 holds only if the 95% CI lower bound of sign accuracy "
                    ">0.5 AND it beats every baseline under all primary LOO splits",
            report_both_regimes=True,
            regimes=["injected (internal validity)", "official det anchor (external validity)"],
            no_result_driven_regime_switch=True,
        ),

        datasets=dict(
            mot17=dict(train_sequences_with_gt=7, det_files_total=42,
                       det_files_calibratable=21, role="main"),
            dancetrack=dict(val_sequences=25, official_det=0, role="main"),
            mot20=dict(train_sequences=4, det_files_total=8,
                       det_files_calibratable=4, role="external validation"),
            correction="design v1 quoted '42 MOT17 / 8 MOT20 det for Theta "
                       "calibration'; those totals include TEST which has no "
                       "public GT. Calibratable counts are 21 and 4.",
        ),

        calibration=dict(
            n_sequences=calib["n_sequences"],
            score_distribution=calib["score_distribution"],
            per_sequence=calib["per_sequence"],
        ),

        quality_gate=dict(
            idsw_remap="src_gid side channel; the old _key(frame, round(box,1)) "
                       "lookup is INVALID under injection (selftest proves it)",
            identity_condition="Theta=0 must reproduce GT boxes exactly",
            determinism="same seed -> identical detections",
        ),

        falsification=dict(
            novel_claim="no retrieved work satisfies all four: (a) orthogonal "
                        "continuous noise calibration sweep, (b) response is "
                        "marginal association gain, (c) formal sign-flip "
                        "boundary, (d) pre-evaluation label-free predictor with LOO",
            if_disproved="withdraw the novelty claim, downgrade to a "
                         "cross-dataset / cross-detector replication+extension",
            search_limitation="'not retrieved' is NOT 'does not exist'",
        ),
    )
    cfg["integrity"] = dict(algorithm="sha256", digest=_hash(cfg))
    return cfg


def verify_frozen(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    got = _hash(cfg)
    want = cfg.get("integrity", {}).get("digest")
    cfg["_verify"] = dict(ok=(got == want), computed=got, stored=want)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["local", "server"], default="local")
    ap.add_argument("--zip-dir", default=r"/path/to/mot_exp_assets\datasets")
    ap.add_argument("--data-root", default="/root/mot_exp/data")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reuse-calibration", action="store_true",
                    help="rebuild the grid from an existing calibration_report.json "
                         "instead of re-reading the 15 GB zips")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, FROZEN_FILE)

    if a.verify_only:
        v = verify_frozen(fp)
        print(json.dumps(v["_verify"], indent=2))
        return 0 if v["_verify"]["ok"] else 2

    if os.path.exists(fp) and not a.force:
        v = verify_frozen(fp)
        print("FROZEN FILE ALREADY EXISTS:", fp)
        print("  hash ok:", v["_verify"]["ok"],
              " conditions:", v.get("n_conditions_unique"))
        print("  refusing to overwrite. Use --force only to RE-FREEZE explicitly.")
        return 3

    print("=== [1/3] Theta-hat calibration from official det.txt ===", flush=True)
    cp = os.path.join(a.out, "calibration_report.json")
    if a.reuse_calibration and os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            old = json.load(f)
        calib = dict(n_sequences=old["n_sequences"],
                     per_sequence=old["per_sequence"],
                     grid=dc.build_grid(old["per_sequence"]),
                     score_distribution=old["score_distribution"],
                     n_scores=old.get("n_scores", 0))
        print("  reused calibration_report.json, rebuilt grid with current rule")
    else:
        if a.source == "local":
            pairs = dc.iter_pairs_from_zips(a.zip_dir)
        else:
            pairs = dc.iter_pairs_from_root(a.data_root)
        calib = dc.calibrate(pairs)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)

    print("\n=== [2/3] Freeze Phase 0 config ===", flush=True)
    cfg = build_frozen(calib)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    v = verify_frozen(fp)
    assert v["_verify"]["ok"], "hash seal failed"
    print("  frozen:", fp)
    print("  sha256:", cfg["integrity"]["digest"])
    print("  unique conditions:", cfg["n_conditions_unique"])
    print("  grid:", json.dumps(cfg["grid"], ensure_ascii=False)[:600])

    print("\n=== [3/3] Injection invariants + IDSW remapping quality gate ===",
          flush=True)
    selftest = st.run()
    with open(os.path.join(a.out, "phase1_selftest.json"), "w",
              encoding="utf-8") as f:
        json.dump(selftest, f, indent=2, ensure_ascii=False)
    for c in selftest["checks"]:
        print("  %-42s %s  %s" % (c["name"], "PASS" if c["passed"] else "FAIL",
                                 c["detail"]))
    print("  GATE:", selftest["quality_gate"])

    report = dict(
        generated_at=_utc(), source=a.source, protocol=PROTOCOL,
        calibration_sequences=calib["n_sequences"],
        grid_rule=calib["grid"]["rule"],
        n_conditions_unique=cfg["n_conditions_unique"],
        frozen_sha256=cfg["integrity"]["digest"],
        quality_gate=selftest["quality_gate"],
        selftest_passed=selftest["passed"],
        checks_passed=sum(1 for c in selftest["checks"] if c["passed"]),
        checks_total=len(selftest["checks"]),
        old_key_tp_unmatched=selftest.get("old_key_tp_unmatched"),
        phase="Phase 0+1 complete; Phase 2 NOT started",
        next_step="user sign-off on frozen grid before any GPU run",
    )
    with open(os.path.join(a.out, "phase1_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n" + json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if selftest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
