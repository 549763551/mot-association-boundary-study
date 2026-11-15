#!/usr/bin/env python3
"""Phase 1 self-test: injection invariants + IDSW identity-remapping gate.

The old analysis code maps a tracker row back to a GT identity with
``_key(frame, round(box, 1))``. That works ONLY when the tracker input boxes
are exactly the GT boxes (the oracle-detection regime of the aborted
experiment). Under noise injection it silently produces unmatched rows and
therefore wrong IDSW. This module proves the failure and validates the
replacement: the ``src_gid`` side channel carried by every injected detection.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from noise_injection import (FP_GID, DetFrame, Theta, inject,
                             requires_feature_reextraction, tp_boxes_unchanged)

IMG = (1920.0, 1080.0)


def make_synth_gt(n_frames: int = 24, n_targets: int = 3
                  ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Three linearly moving targets; target 2 is occluded on frames 8..10."""
    gt = {}
    for f in range(1, n_frames + 1):
        boxes, gids = [], []
        for i in range(1, n_targets + 1):
            if i == 2 and 8 <= f <= 10:
                continue
            x = 120.0 + 90.0 * i + 7.0 * f + 3.0 * i * f
            y = 200.0 + 55.0 * i + 2.0 * f
            w = 60.0 + 10.0 * i
            h = 150.0 + 20.0 * i
            boxes.append([x, y, w, h])
            gids.append(i)
        gt[f] = (np.asarray(boxes, dtype=np.float64),
                 np.asarray(gids, dtype=np.int64))
    return gt


def _key(frame, box):
    """Exact copy of full_experiment/analyze.py::_key."""
    return (int(frame), tuple(np.round(np.asarray(box, dtype=float), 1)))


def build_gt_key(gt) -> dict:
    out = {}
    for f, (boxes, gids) in gt.items():
        for b, g in zip(boxes, gids):
            out[_key(f, b)] = int(g)
    return out


def identity_rows(det_frames, swap_after: int = -1):
    """A perfect tracker: tid == src_gid. Optionally swap two identities."""
    rows = []
    for f in sorted(det_frames):
        D = det_frames[f]
        for j in range(len(D)):
            gid = int(D.src_gid[j])
            tid = gid if gid != FP_GID else -(1000 + j)
            if swap_after > 0 and f >= swap_after and gid in (1, 2):
                tid = 2 if gid == 1 else 1
            rows.append((f, tid, j, D.boxes[j].copy()))
    return rows


def idsw_from_rows(det_frames, rows, gt_key, use_src_gid: bool = True):
    """Count per-frame ID switches; return (switches, unmatched, tp_unmatched).

    ``tp_unmatched`` counts rows that really came from a GT box (known via the
    side channel) but that the OLD box-key mapping failed to resolve. It is the
    decisive evidence that the old remapping is invalid under injection.
    """
    by_frame: Dict[int, list] = {}
    for r in rows:
        by_frame.setdefault(r[0], []).append(r)
    prev, switches, unmatched, tp_unmatched = {}, 0, 0, 0
    for f in sorted(by_frame):
        cur = {}
        D = det_frames[f]
        for (fr, tid, det_idx, box) in by_frame[f]:
            true_gid = int(D.src_gid[det_idx]) if det_idx < len(D.src_gid) else FP_GID
            if use_src_gid:
                gid = true_gid
            else:
                gid = gt_key.get(_key(fr, box), FP_GID)
                if gid == FP_GID and true_gid != FP_GID:
                    tp_unmatched += 1
            if gid == FP_GID:
                unmatched += 1
                continue
            cur[gid] = tid
        n = 0
        for gid, tid in cur.items():
            if gid in prev and prev[gid] != tid:
                n += 1
        switches += n
        prev.update(cur)
    return int(switches), int(unmatched), int(tp_unmatched)


def run() -> dict:
    gt = make_synth_gt()
    checks: List[dict] = []

    def check(name, ok, detail):
        checks.append(dict(name=name, passed=bool(ok), detail=detail))

    # --- 1. identity condition ------------------------------------------
    zero = Theta(0.0, 0.0, 0.0, 1)
    d0, s0 = inject(gt, zero, 7, IMG)
    same_boxes = all(np.array_equal(d0[f].boxes.astype(np.float64), gt[f][0])
                     for f in gt)
    same_ids = all(np.array_equal(d0[f].src_gid, gt[f][1]) for f in gt)
    check("identity_boxes", same_boxes, "Theta=0 reproduces GT boxes exactly")
    check("identity_ids", same_ids, "Theta=0 reproduces GT identities exactly")
    check("identity_no_fp_fn", s0["n_fp"] == 0 and s0["n_fn"] == 0,
          "n_fp=%d n_fn=%d" % (s0["n_fp"], s0["n_fn"]))

    # --- 2. determinism ---------------------------------------------------
    th = Theta(0.05, 0.10, 0.10, 3)
    d1, st1 = inject(gt, th, 12345, IMG)
    d2, st2 = inject(gt, th, 12345, IMG)
    det = all(np.array_equal(d1[f].boxes, d2[f].boxes) and
              np.array_equal(d1[f].src_gid, d2[f].src_gid) and
              np.array_equal(d1[f].scores, d2[f].scores) for f in gt)
    check("determinism", det, "same seed -> byte-identical detections")
    d3, _ = inject(gt, th, 999, IMG)
    differs = any(not np.array_equal(d1[f].boxes, d3[f].boxes) for f in gt)
    check("seed_sensitivity", differs, "different seed -> different detections")

    # --- 3. monotonicity of beta / lambda on a large sample ---------------
    big = make_synth_gt(n_frames=400, n_targets=6)
    rates_b, rates_l = [], []
    for b in (0.0, 0.1, 0.3, 0.6):
        _, s = inject(big, Theta(0.0, b, 0.0, 1), 5, IMG)
        rates_b.append(s["observed_fn_rate"])
    for l in (0.0, 0.1, 0.3, 0.6):
        _, s = inject(big, Theta(0.0, 0.0, l, 1), 5, IMG)
        rates_l.append(s["observed_fp_rate"])
    check("beta_monotone", all(rates_b[i] <= rates_b[i + 1] + 1e-9
                               for i in range(3)), str([round(v, 3) for v in rates_b]))
    check("lambda_monotone", all(rates_l[i] <= rates_l[i + 1] + 1e-9
                                 for i in range(3)), str([round(v, 3) for v in rates_l]))
    check("beta_calibrated", abs(rates_b[2] - 0.3) < 0.06,
          "observed FN rate %.3f vs nominal 0.300" % rates_b[2])
    check("lambda_calibrated", abs(rates_l[2] - 0.3) < 0.06,
          "observed FP rate %.3f vs nominal 0.300" % rates_l[2])

    # --- 3b. feature-cache structural rule --------------------------------
    th_bt = Theta(0.0, 0.30, 0.20, 5)
    d_bt, _ = inject(gt, th_bt, 42, IMG)
    check("beta_lambda_tau_reuse_gt_features",
          tp_boxes_unchanged(gt, d_bt) and not requires_feature_reextraction(th_bt),
          "surviving TP boxes are bit-identical to GT => cached features reusable")
    check("sigma_loc_forces_reextraction",
          requires_feature_reextraction(th) and not tp_boxes_unchanged(gt, d1),
          "sigma_loc>0 moves boxes => re-extraction genuinely required")

    # --- 4. scale normalization ------------------------------------------
    small = {1: (np.asarray([[100, 100, 30, 75]], dtype=np.float64),
                 np.asarray([1], dtype=np.int64))}
    large = {1: (np.asarray([[100, 100, 60, 150]], dtype=np.float64),
                 np.asarray([1], dtype=np.int64))}
    ds, _ = inject(small, Theta(0.10, 0, 0, 1), 3, IMG)
    dl, _ = inject(large, Theta(0.10, 0, 0, 1), 3, IMG)
    off_s = abs(float(ds[1].boxes[0, 0] - 100))
    off_l = abs(float(dl[1].boxes[0, 0] - 100))
    check("scale_normalized", off_l > off_s,
          "same sigma_loc, |dx| small=%.3f large=%.3f (should scale with h)" %
          (off_s, off_l))

    # --- 5. IDSW remapping quality gate ----------------------------------
    gt_key = build_gt_key(gt)
    rows_ok = identity_rows(d0)
    self_zero, _, _ = idsw_from_rows(d0, rows_ok, gt_key, use_src_gid=True)
    old_zero, old_unmatched0, old_tp0 = idsw_from_rows(
        d0, rows_ok, gt_key, use_src_gid=False)
    check("oracle_old_key_valid", old_unmatched0 == 0 and old_zero == 0 and old_tp0 == 0,
          "no injection: old box-key mapping gives 0 switches, %d unmatched" %
          old_unmatched0)

    rows_new = identity_rows(d1)
    sw_new, unmatched_new, _ = idsw_from_rows(d1, rows_new, gt_key, use_src_gid=True)
    check("src_gid_perfect_tracker_zero_idsw", sw_new == 0,
          "perfect tracker under injection -> %d IDSW" % sw_new)
    sw_old, unmatched_old, tp_unmatched = idsw_from_rows(
        d1, rows_new, gt_key, use_src_gid=False)
    check("old_key_breaks_under_injection", tp_unmatched > 0,
          "old box-key mapping fails on %d TRUE-POSITIVE rows (must be > 0); "
          "total unmatched=%d, old-method IDSW=%d" %
          (tp_unmatched, unmatched_old, sw_old))

    rows_swap = identity_rows(d0, swap_after=12)
    sw_swap, _, _ = idsw_from_rows(d0, rows_swap, gt_key, use_src_gid=True)
    check("idsw_detects_injected_switches", sw_swap == 2,
          "2 deliberately swapped identities -> %d IDSW" % sw_swap)
    sw_swap_new, _, _ = idsw_from_rows(d1, identity_rows(d1, swap_after=12),
                                       gt_key, use_src_gid=True)
    check("idsw_detects_switches_under_injection", sw_swap_new == 2,
          "same swap under injection -> %d IDSW" % sw_swap_new)

    passed = all(c["passed"] for c in checks)
    return dict(quality_gate="IDSW_QUALITY_GATE_OK" if passed else "IDSW_QUALITY_GATE_FAILED",
                passed=passed, checks=checks,
                old_key_tp_unmatched=int(tp_unmatched),
                injected_stats=st1,
                note="src_gid is a side channel; trackers never see GT identities at run time")


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, ensure_ascii=False))
