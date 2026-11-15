#!/usr/bin/env python3
"""Local CPU self-test for the Phase 2 pipeline.

Runs entirely offline: toy detections, deterministic pseudo features, and an
in-memory extraction of upstream TrackEval for a NUMERICAL cross-check of the
AssA implementation. No server, no GPU, no traffic, no real DeltaAssA observed.
"""
from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile

import numpy as np

from noise_injection import Theta, inject, grid_conditions, requires_feature_reextraction
from phase2_core import (DetSeq, xywh_to_xyxy, run_tracker, hota_metrics,
                         idsw_sidechannel, idf1_sidechannel,
                         observable_features, STRATEGIES, OBSERVABLES,
                         iou_matrix)
from phase2_features import FeatureBank, ImageCache
from run_phase2 import load_frozen, make_detseq, evaluate_strategy, HERE

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(dict(name=name, passed=bool(ok), detail=str(detail)))
    print("  %-38s %s  %s" % (name, "OK" if ok else "FAIL", detail), flush=True)


def toy_gt(n_frames=30, n_targets=6):
    gt = {}
    for f in range(1, n_frames + 1):
        boxes, gids = [], []
        for g in range(1, n_targets + 1):
            x = 100 + g * 120 + 3 * f
            y = 100 + (g % 3) * 200 + 2 * f
            boxes.append([float(x), float(y), 60.0, 120.0])
            gids.append(g)
        gt[f] = (np.asarray(boxes, dtype=np.float64),
                 np.asarray(gids, dtype=np.int64))
    return gt


def toy_seqdata(gt):
    xyxy = {f: xywh_to_xyxy(b) for f, (b, _) in gt.items()}
    return dict(frames=sorted(gt), gt_for_inject=gt,
                gt_eval={f: (xyxy[f], gt[f][1]) for f in gt},
                gt_xywh={f: gt[f][0] for f in gt},
                gt_gids={f: gt[f][1] for f in gt},
                gt_xyxy=xyxy, frame_paths={}, img_size=(1920.0, 1080.0),
                n_frames_total=len(gt), dir="toy")


def official_assa(gt_eval, det_xyxy, tids_eval):
    """Call upstream trackeval.metrics.HOTA.eval_sequence in memory."""
    tar = os.environ.get("MOT_TRACKEVAL_TAR",
                         r"/path/to/mot_exp_assets\code\TrackEval.tar.gz")
    if not os.path.exists(tar):
        return None, None, "TrackEval.tar.gz not found"
    tmp = tempfile.mkdtemp(prefix="te_selftest_")
    with tarfile.open(tar, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.startswith("TrackEval-master/trackeval/"):
                tf.extract(m, tmp, filter="data")
    sys.path.insert(0, os.path.join(tmp, "TrackEval-master"))
    for n, t in (("float", float), ("int", int), ("bool", bool),
                 ("object", object), ("str", str)):
        if not hasattr(np, n):
            setattr(np, n, t)
    import trackeval
    frames = sorted(gt_eval)
    gids = sorted({int(g) for f in frames for g in gt_eval[f][1]})
    tids = sorted({int(t) for f in frames for t in tids_eval[f]})
    gm = {g: i for i, g in enumerate(gids)}
    tm = {t: i for i, t in enumerate(tids)}
    gt_ids, tr_ids, sims = [], [], []
    for f in frames:
        G, gr = gt_eval[f]
        D = det_xyxy[f]
        T = tids_eval[f]
        gt_ids.append(np.asarray([gm[int(g)] for g in gr], dtype=np.int64))
        tr_ids.append(np.asarray([tm[int(t)] for t in T], dtype=np.int64))
        sims.append(iou_matrix(G, D))
    data = dict(
        num_gt_dets=sum(len(x) for x in gt_ids),
        num_tracker_dets=sum(len(x) for x in tr_ids),
        num_gt_ids=len(gm), num_tracker_ids=len(tm),
        gt_ids=gt_ids, tracker_ids=tr_ids, similarity_scores=sims,
        num_timesteps=len(frames), seq="toy")
    res = trackeval.metrics.HOTA().eval_sequence(data)
    return (float(np.mean(res["AssA"])), float(np.mean(res["DetA"])),
            "upstream HOTA.eval_sequence")


def main():
    print("=== Phase 2 local self-test (offline, toy) ===", flush=True)
    cfg = load_frozen(os.path.join(HERE, "frozen_config.json"))
    conds = grid_conditions(cfg["grid"])
    check("frozen hash seal", cfg["integrity"]["digest"].startswith("71dc864e"),
          cfg["integrity"]["digest"][:16])
    check("condition enumeration", len(conds) == 28, "n=%d" % len(conds))
    check("geometry-keyed reextract count",
          sum(1 for t in conds if requires_feature_reextraction(t)) == 9,
          "9 of 28 conditions need re-extraction")

    gt = toy_gt()
    sd = toy_seqdata(gt)
    bank = FeatureBank(sd, ImageCache(sd), backend="toy")

    # --- identity / determinism -----------------------------------------
    d0, s0 = inject(sd["gt_for_inject"], Theta(0, 0, 0, 1), 7, sd["img_size"])
    exact = all(
        np.array_equal(d0[f].boxes[j].astype(np.float64),
                       {int(g): b for b, g in
                        zip(gt[f][0], gt[f][1])}[int(d0[f].src_gid[j])])
        for f in d0 for j in range(len(d0[f])))
    check("Theta=0 reproduces GT boxes", exact,
          "n_gt=%d n_tp=%d" % (s0["n_gt"], s0["n_tp"]))
    check("Theta=0 has no FP/FN",
          s0["n_fp"] == 0 and s0["n_fn"] == 0,
          "fp=%d fn=%d" % (s0["n_fp"], s0["n_fn"]))
    d0b, _ = inject(sd["gt_for_inject"], Theta(0, 0, 0, 1), 7, sd["img_size"])
    check("same seed -> identical detections",
          all(np.array_equal(d0[f].boxes, d0b[f].boxes) and
              np.array_equal(d0[f].src_gid, d0b[f].src_gid) for f in d0))

    # --- a noisy condition ------------------------------------------------
    theta = Theta(0.05, 0.20, 0.20, 10)
    d1, s1 = inject(sd["gt_for_inject"], theta, 7, sd["img_size"])
    check("injection changes the detection set",
          s1["n_tp"] < s1["n_gt"] and s1["n_fp"] > 0,
          "gt=%d tp=%d fp=%d fn=%d" %
          (s1["n_gt"], s1["n_tp"], s1["n_fp"], s1["n_fn"]))
    check("sigma_loc forces re-extraction",
          requires_feature_reextraction(theta) and
          not requires_feature_reextraction(Theta(0, .2, .2, 10)))

    feats = bank.det_features_for(d1, theta)
    ds = make_detseq(d1, feats)
    check("feature matrix shape",
          all(ds.feats[f].shape == (len(d1[f]), 512) for f in ds.frames))
    obs = observable_features(ds)
    check("observables are <=8 label-free dims",
          len(obs) == 8 and tuple(obs) == OBSERVABLES and
          all(np.isfinite(v) for v in obs.values()))

    # --- six strategies ---------------------------------------------------
    results = {}
    ok_all = True
    for s in STRATEGIES:
        tids, st = run_tracker(s, ds, seed=7)
        shape_ok = all(len(tids[f]) == len(d1[f]) for f in ds.frames)
        m = evaluate_strategy(sd, d1, ds, s, 7)
        rng_ok = (0.0 <= m["AssA"] <= 1.0 and 0.0 <= m["DetA"] <= 1.0
                  and 0.0 <= m["HOTA"] <= 1.0 and 0.0 <= m["IDF1"] <= 1.0
                  and m["IDSW"] >= 0)
        ok_all &= bool(shape_ok and rng_ok)
        results[s] = m
        print("      %-3s AssA=%.4f DetA=%.4f IDSW=%4d IDF1=%.4f "
              "abstain=%d reid=%d" %
              (s, m["AssA"], m["DetA"], m["IDSW"], m["IDF1"],
               m["n_abstain"], m["reid_calls"]), flush=True)
    check("all 6 strategies run + metrics in range", ok_all)
    # A1 (ByteTrack) deliberately discards low-score boxes, so its output det
    # set -- and hence DetA -- legitimately differs. All other strategies keep
    # the identical injected detection set, which is what makes DeltaAssA a
    # clean association-only contrast (design §4.3).
    same_det_set = [s for s in STRATEGIES if s != "A1"]
    detas = [results[s]["DetA"] for s in same_det_set]
    check("DetA identical for same-det-set strategies",
          max(detas) - min(detas) < 1e-9,
          "%s spread=%.2e (A1 excluded by design)" %
          ("/".join(same_det_set), max(detas) - min(detas)))
    check("A1 low-score filtering is active",
          results["A1"]["n_output_dets"] < results["A0"]["n_output_dets"],
          "A1=%d A0=%d" % (results["A1"]["n_output_dets"],
                           results["A0"]["n_output_dets"]))
    check("main response computable",
          abs(results["A4"]["AssA"] - results["A0"]["AssA"]) < 1.0,
          "dAssA(A4-A0)=%+.4f" %
          (results["A4"]["AssA"] - results["A0"]["AssA"]))

    # --- IDSW side channel -------------------------------------------------
    tids0, _ = run_tracker("A0", ds, seed=7)
    sw = idsw_sidechannel(d1, tids0)
    check("src_gid IDSW is a non-negative integer", isinstance(sw, int) and sw >= 0,
          "IDSW=%d" % sw)

    # --- official TrackEval numerical cross-check ---------------------------
    tids_e, _ = run_tracker("A2", ds, seed=7)
    keep = {f: tids_e[f] >= 0 for f in ds.frames}
    det_xyxy = {f: ds.xyxy[f][keep[f]] for f in ds.frames}
    tids_eval = {f: tids_e[f][keep[f]] for f in ds.frames}
    mine = hota_metrics(sd["gt_eval"], det_xyxy, tids_eval)
    off, off_deta, note = official_assa(sd["gt_eval"], det_xyxy, tids_eval)
    if off is None:
        check("TrackEval AssA cross-check", False, note)
        check("TrackEval DetA cross-check", False, note)
    else:
        check("TrackEval AssA cross-check", abs(mine["AssA"] - off) < 1e-6,
              "mine=%.8f official=%.8f diff=%.2e" %
              (mine["AssA"], off, abs(mine["AssA"] - off)))
        check("TrackEval DetA cross-check", abs(mine["DetA"] - off_deta) < 1e-6,
              "mine=%.8f official=%.8f diff=%.2e" %
              (mine["DetA"], off_deta, abs(mine["DetA"] - off_deta)))

    passed = all(c["passed"] for c in CHECKS)
    out = dict(status="PHASE2_SELFTEST_OK" if passed else "PHASE2_SELFTEST_FAILED",
               passed=passed, checks=CHECKS, n_checks=len(CHECKS),
               n_passed=sum(1 for c in CHECKS if c["passed"]),
               official_assa=off, mine_assa=mine["AssA"])
    p = os.path.join(HERE, "phase2_selftest.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("=== %s  %d/%d ===" %
          (out["status"], out["n_passed"], out["n_checks"]), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
