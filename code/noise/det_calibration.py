#!/usr/bin/env python3
"""Reverse-engineer Theta-hat from official MOTChallenge det.txt (design §3.1).

For every TRAIN sequence with both gt.txt and det.txt we match detections to GT
boxes with Hungarian + IoU>=0.5 and estimate:

  sigma_loc : robust spread (MAD around median) of centre offsets dx/h, dy/h
  beta      : 1 - matched_GT / GT                    (miss rate)
  lambda    : (det - matched_det) / GT              (false positives per GT)
  tau       : mean run length of consecutive missed frames for the same GT id

Only TRAIN detections can be calibrated (TEST has no public GT). This is a
correction to design v1, which quoted "42 MOT17 / 8 MOT20 det files": those
numbers include TEST; the calibratable counts are 21 and 4.

The resulting quantiles anchor the frozen noise grid to real detector quality,
so the phrase "equivalent to a real detector" is verifiable rather than a guess.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from typing import Dict, Iterator, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

GT_CONF_COL, GT_CLASS_COL, PEDESTRIAN_CLASS = 6, 7, 1
IOU_MATCH = 0.5
MAD_SCALE = 1.4826


def read_mot(raw: bytes) -> np.ndarray:
    txt = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
    return np.loadtxt(io.StringIO(txt), delimiter=",", ndmin=2)


def filter_gt(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] <= GT_CLASS_COL:
        return arr
    keep = (arr[:, GT_CONF_COL] != 0) & (arr[:, GT_CLASS_COL] == PEDESTRIAN_CLASS)
    return arr[keep]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a,b: (N,4)/(M,4) xywh -> IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax = np.stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]], 1)
    bx = np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)
    ix1 = np.maximum(ax[:, None, 0], bx[None, :, 0])
    iy1 = np.maximum(ax[:, None, 1], bx[None, :, 1])
    ix2 = np.minimum(ax[:, None, 2], bx[None, :, 2])
    iy2 = np.minimum(ax[:, None, 3], bx[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    aa = np.maximum(0.0, ax[:, 2] - ax[:, 0]) * np.maximum(0.0, ax[:, 3] - ax[:, 1])
    bb = np.maximum(0.0, bx[:, 2] - bx[:, 0]) * np.maximum(0.0, bx[:, 3] - bx[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 3:
        return float(np.std(v)) if len(v) else 0.0
    return float(MAD_SCALE * np.median(np.abs(v - np.median(v))))


def calibrate_one(gt_arr: np.ndarray, det_arr: np.ndarray) -> dict:
    gt_arr = filter_gt(gt_arr)
    det_arr = det_arr[:, :7] if det_arr.shape[1] >= 7 else det_arr
    gt_by: Dict[int, list] = {}
    for r in gt_arr:
        gt_by.setdefault(int(r[0]), []).append(r)
    det_by: Dict[int, list] = {}
    for r in det_arr:
        det_by.setdefault(int(r[0]), []).append(r)

    n_gt = n_det = n_match = 0
    dxn, dyn, dwn, dhn = [], [], [], []
    present: Dict[int, List[Tuple[int, bool]]] = {}
    all_scores: List[float] = []
    for f in sorted(set(gt_by) | set(det_by)):
        G = np.asarray(gt_by.get(f, []), dtype=np.float64).reshape(-1, gt_arr.shape[1])
        D = np.asarray(det_by.get(f, []), dtype=np.float64).reshape(-1, det_arr.shape[1])
        gb = G[:, 2:6] if len(G) else np.zeros((0, 4))
        db = D[:, 2:6] if len(D) else np.zeros((0, 4))
        gids = G[:, 1].astype(np.int64) if len(G) else np.zeros((0,), np.int64)
        if len(D):
            all_scores.extend(D[:, 6].tolist())
        n_gt += len(gb)
        n_det += len(db)
        m_gt, m_det = set(), set()
        if len(gb) and len(db):
            iou = iou_matrix(gb, db)
            ri, di = linear_sum_assignment(1.0 - iou)
            ok = iou[ri, di] >= IOU_MATCH
            for r, d in zip(ri[ok], di[ok]):
                m_gt.add(int(r))
                m_det.add(int(d))
                gx, gy, gw, gh = gb[r]
                dx, dy, dw, dh = db[d]
                hh = max(1.0, float(gh))
                ww = max(1.0, float(gw))
                dxn.append((dx + dw / 2 - gx - gw / 2) / hh)
                dyn.append((dy + dh / 2 - gy - gh / 2) / hh)
                dwn.append((dw - gw) / ww)
                dhn.append((dh - gh) / hh)
        n_match += len(m_gt)
        for i, g in enumerate(gids):
            present.setdefault(int(g), []).append((f, i in m_gt))

    runs: List[int] = []
    for _, obs in present.items():
        cur = 0
        last_f = None
        for f, matched in obs:
            if last_f is not None and f != last_f + 1:
                if cur:
                    runs.append(cur)
                cur = 0
            if matched:
                if cur:
                    runs.append(cur)
                cur = 0
            else:
                cur += 1
            last_f = f
        if cur:
            runs.append(cur)

    sx, sy = _mad(dxn), _mad(dyn)
    rmse = float(np.sqrt(np.mean(np.asarray(dxn) ** 2 + np.asarray(dyn) ** 2) / 2)) \
        if dxn else 0.0
    return dict(
        n_gt=int(n_gt), n_det=int(n_det), n_match=int(n_match),
        sigma_loc=float(np.sqrt(0.5 * (sx * sx + sy * sy))),
        sigma_loc_rmse=rmse,
        sigma_loc_mad_x=sx, sigma_loc_mad_y=sy,
        scale_mad_w=_mad(dwn), scale_mad_h=_mad(dhn),
        beta=float(1.0 - n_match / max(1, n_gt)),
        lambd=float((n_det - n_match) / max(1, n_gt)),
        tau=float(np.mean(runs)) if runs else 1.0,
        tau_p90=float(np.quantile(runs, 0.9)) if runs else 1.0,
        n_runs=len(runs),
        score_mean=float(np.mean(all_scores)) if all_scores else None,
        scores=all_scores,
    )


def iter_pairs_from_zips(zip_dir: str) -> Iterator[Tuple[str, bytes, bytes]]:
    """Yield (label, gt_bytes, det_bytes) for TRAIN sequences in local zips."""
    for zp, prefix in (("MOT17.zip", "MOT17/train/"),
                       ("MOT20.zip", "MOT20/train/")):
        path = os.path.join(zip_dir, zp)
        if not os.path.exists(path):
            print("  WARN missing", path, file=sys.stderr)
            continue
        with zipfile.ZipFile(path) as z:
            dets = [n for n in z.namelist()
                    if n.startswith(prefix) and n.endswith("/det/det.txt")]
            for d in sorted(dets):
                g = d.rsplit("/", 2)[0] + "/gt/gt.txt"
                if g in z.namelist():
                    yield os.path.basename(zp) + ":" + d, z.read(g), z.read(d)


def iter_pairs_from_root(root: str) -> Iterator[Tuple[str, bytes, bytes]]:
    for base in (os.path.join(root, "mot17/MOT17/train"),
                 os.path.join(root, "mot20/MOT20/train")):
        if not os.path.isdir(base):
            continue
        for seq in sorted(os.listdir(base)):
            d = os.path.join(base, seq, "det", "det.txt")
            g = os.path.join(base, seq, "gt", "gt.txt")
            if os.path.isfile(d) and os.path.isfile(g):
                yield seq, open(g, "rb").read(), open(d, "rb").read()


def _quantiles(vals: List[float]) -> dict:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {f"p{int(q * 100)}": 0.0
                for q in (0.10, 0.25, 0.50, 0.75, 0.90)}
    return {f"p{int(q * 100)}": float(np.quantile(v, q))
            for q in (0.10, 0.25, 0.50, 0.75, 0.90)}


def build_grid(per_seq: List[dict]) -> dict:
    """FROZEN grid rule (hybrid_v1).

    Pure quantile anchoring was rejected during Phase 1: official det has
    median beta ~0.43, so quantiles alone leave the whole 0 -> 0.39 low-noise
    region unsampled, which is exactly where a positive-gain window would sit.
    The frozen rule therefore HYBRIDIZES, before any result is observed:
      levels = sorted(unique({0, design-v1 low levels, one Theta-hat quantile,
                              pre-declared physical upper bound}))
    Design-v1 low levels (0.02/0.05/0.10 and 0.05/0.10/0.20) are pre-declared
    in EXPERIMENT_DESIGN_NOISE_GRADIENT.md §4.2; the quantile ties the grid to
    real detector quality; the upper bound guarantees the sweep crosses any
    sign-flip boundary. No level is chosen from an observed DeltaAssA value.
    """
    s = _quantiles([r["sigma_loc"] for r in per_seq])
    b = _quantiles([r["beta"] for r in per_seq])
    l = _quantiles([r["lambd"] for r in per_seq])
    t = _quantiles([r["tau"] for r in per_seq])

    def levels(low, quantile, cap):
        raw = [0.0] + list(low) + [float(quantile), float(cap)]
        out = []
        for v in sorted(round(float(min(cap, max(0.0, v))), 6) for v in raw):
            if not out or v > out[-1] + 1e-6:
                out.append(v)
        return out

    tau_levels = sorted({1, max(1, int(round(t["p50"]))),
                         max(2, int(round(t["p90"])))})
    grid = dict(
        rule="hybrid_v1",
        caps=dict(sigma_loc=0.20, beta=0.75, lambd=0.50),
        rule_note="0 + design-v1 low levels + one Theta-hat quantile + "
                  "pre-declared upper bound; frozen before any GPU result",
        rejection_note="pure quantile anchoring rejected: would skip 0..0.39 "
                       "beta where the gain window is most likely",
        quantiles=dict(sigma_loc=s, beta=b, lambd=l, tau=t),
        sigma_loc=levels([0.02, 0.05, 0.10], s["p75"], 0.20),
        beta=levels([0.05, 0.10, 0.20], b["p50"], 0.75),
        lambd=levels([0.05, 0.10, 0.20], l["p90"], 0.50),
        tau=tau_levels,
        interaction=dict(
            sigma_loc=[0.05, 0.20],
            beta=[0.20, round(float(b["p50"]), 6)],
            lambd=[0.20, round(float(l["p90"]), 6)],
            tau=[1, max(2, int(round(t["p90"])))],
        ),
        interaction_note="interaction cells deliberately pair a moderate level "
                         "with a high/real-detector level so interactions are "
                         "observable; pre-declared, not fitted",
    )
    return grid


def calibrate(pairs: Iterator[Tuple[str, bytes, bytes]], verbose: bool = True) -> dict:
    per_seq, all_scores = [], []
    for label, graw, draw in pairs:
        try:
            r = calibrate_one(read_mot(graw), read_mot(draw))
        except Exception as e:
            print("  FAIL", label, type(e).__name__, e, file=sys.stderr)
            continue
        r["label"] = label
        sc = r.pop("scores")
        all_scores.extend(sc)
        per_seq.append(r)
        if verbose:
            print("  %-58s s=%.4f b=%.3f l=%.3f t=%.2f  gt=%d det=%d" %
                  (label[:58], r["sigma_loc"], r["beta"], r["lambd"],
                   r["tau"], r["n_gt"], r["n_det"]), flush=True)
    if not per_seq:
        raise RuntimeError("no calibratable (train gt + det) pairs found")
    grid = build_grid(per_seq)
    return dict(n_sequences=len(per_seq), per_sequence=per_seq,
                grid=grid, score_distribution=_quantiles(all_scores),
                n_scores=len(all_scores))
