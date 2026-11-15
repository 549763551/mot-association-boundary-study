#!/usr/bin/env python3
"""Calibrated orthogonal detection-noise injection (pre-registered, design §3.1).

Theta = (sigma_loc, beta, lambda, tau)
  sigma_loc : Gaussian jitter of box centre and scale, normalized by target height
  beta      : per-target per-frame deletion probability (false negatives)
  lambda    : expected false positives per ground-truth box per frame
  tau       : temporal correlation length L; a track keeps the same failure
              state and the same jitter for L consecutive frames (L=1 => i.i.d.)

Every injected detection carries ``src_gid``: the GT identity it came from, or
-1 for a false positive. This is the ONLY sanctioned route for mapping tracker
rows back to GT identities once boxes no longer coincide with GT boxes. The old
``_key(frame, box)`` lookup in full_experiment/analyze.py is invalid under
injection; phase1_selftest.py proves that explicitly.

No GT identity is ever exposed to a tracker at run time. ``src_gid`` lives in a
separate side channel and is consumed only by post-hoc analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FP_GID = -1
_MIN_BOX = 1.0


@dataclass(frozen=True)
class Theta:
    sigma_loc: float = 0.0
    beta: float = 0.0
    lambd: float = 0.0
    tau: int = 1

    def as_dict(self) -> dict:
        return dict(sigma_loc=float(self.sigma_loc), beta=float(self.beta),
                    lambd=float(self.lambd), tau=int(self.tau))

    def key(self) -> str:
        return "s%.6f_b%.6f_l%.6f_t%d" % (self.sigma_loc, self.beta,
                                          self.lambd, self.tau)

    @staticmethod
    def from_dict(d: dict) -> "Theta":
        return Theta(float(d["sigma_loc"]), float(d["beta"]),
                     float(d["lambd"]), int(d["tau"]))


@dataclass
class DetFrame:
    frame: int
    boxes: np.ndarray       # (M,4) xywh, float32
    scores: np.ndarray      # (M,) float32
    src_gid: np.ndarray     # (M,) int64; FP_GID for false positives

    def xyxy(self) -> np.ndarray:
        b = self.boxes.copy()
        b[:, 2] += b[:, 0]
        b[:, 3] += b[:, 1]
        return b

    def __len__(self) -> int:
        return len(self.boxes)


class EmpiricalScores:
    """Draw realistic detection confidences learned from official det.txt.

    True positives sample from the high part of the empirical distribution,
    false positives from the low part. MOT20 det.txt carries a constant score
    of 1.0; in that case a fixed Beta fallback is used.
    """

    def __init__(self, values: Optional[Sequence[float]] = None):
        v = np.asarray(values if values is not None else [], dtype=np.float64)
        v = v[np.isfinite(v)]
        if len(v) < 10 or float(np.std(v)) < 1e-6:
            self.mode = "beta_fallback"
            self.tp = self.fp = None
        else:
            self.mode = "empirical"
            q = np.quantile(v, [0.10, 0.30, 0.50, 1.0])
            self.fp = v[(v >= q[0]) & (v <= q[1])]
            self.tp = v[(v >= q[2]) & (v <= q[3])]
            if len(self.fp) < 5:
                self.fp = v[v <= q[1]]
            if len(self.tp) < 5:
                self.tp = v[v >= q[2]]

    def draw_tp(self, rng, n: int) -> np.ndarray:
        if self.mode == "empirical" and len(self.tp):
            return rng.choice(self.tp, size=n).astype(np.float32)
        return rng.beta(8.0, 2.0, size=n).astype(np.float32)

    def draw_fp(self, rng, n: int) -> np.ndarray:
        if self.mode == "empirical" and len(self.fp):
            return rng.choice(self.fp, size=n).astype(np.float32)
        return rng.beta(2.0, 5.0, size=n).astype(np.float32)


def _clip_box(b: np.ndarray, w_img: float, h_img: float) -> np.ndarray:
    """Clip xywh into the image and keep a minimally valid size."""
    x, y, w, h = b
    w = max(_MIN_BOX, float(w))
    h = max(_MIN_BOX, float(h))
    x = float(np.clip(x, 0.0, max(0.0, w_img - w)))
    y = float(np.clip(y, 0.0, max(0.0, h_img - h)))
    return np.array([x, y, w, h], dtype=np.float32)


def inject(gt: Dict[int, Tuple[np.ndarray, np.ndarray]],
           theta: Theta,
           seed: int,
           img_size: Tuple[float, float],
           scores: Optional[EmpiricalScores] = None
           ) -> Tuple[Dict[int, DetFrame], dict]:
    """Inject calibrated noise into a GT sequence.

    gt: {frame: (boxes(N,4) xywh, gids(N,) int64)}. GT boxes must already be
        filtered with the TrackEval pedestrian convention (see det_calibration).
    Returns ({frame: DetFrame}, stats). Fully deterministic in ``seed``.
    """
    if theta.tau < 1:
        raise ValueError("tau (L) must be >= 1")
    if not (0.0 <= theta.beta <= 1.0) or theta.lambd < 0 or theta.sigma_loc < 0:
        raise ValueError("invalid Theta range")
    rng = np.random.default_rng(int(seed))
    sc = scores or EmpiricalScores()
    w_img, h_img = float(img_size[0]), float(img_size[1])
    frames = sorted(gt)
    if not frames:
        return {}, dict(n_frames=0, n_gt=0, n_tp=0, n_fp=0, n_fn=0)

    f0 = frames[0]
    gids = np.unique(np.concatenate([np.asarray(gt[f][1]) for f in frames]))
    gid_to_row = {int(g): i for i, g in enumerate(gids)}
    n_gid = len(gids)
    n_blocks = (frames[-1] - f0) // theta.tau + 1

    # Per (track, block) latent state: one deletion flag and one jitter vector.
    # Keeping them constant inside a block is what makes failures correlated.
    keep = rng.random((n_gid, n_blocks)) >= theta.beta
    if theta.sigma_loc > 0:
        jitter = rng.normal(0.0, theta.sigma_loc, size=(n_gid, n_blocks, 4))
    else:
        jitter = np.zeros((n_gid, n_blocks, 4), dtype=np.float64)

    out: Dict[int, DetFrame] = {}
    n_gt = n_tp = n_fp = n_fn = 0
    for f in frames:
        boxes, ids = gt[f]
        boxes = np.asarray(boxes, dtype=np.float64)
        ids = np.asarray(ids, dtype=np.int64)
        block = (f - f0) // theta.tau
        n_gt += len(boxes)

        kept_b, kept_g = [], []
        for b, g in zip(boxes, ids):
            r = gid_to_row[int(g)]
            if not keep[r, block]:
                n_fn += 1
                continue
            j = jitter[r, block]
            x, y, w, h = b
            hh = max(_MIN_BOX, float(h))
            nb = np.array([x + j[0] * hh,
                           y + j[1] * hh,
                           w * math.exp(j[2]),
                           h * math.exp(j[3])], dtype=np.float64)
            kept_b.append(_clip_box(nb, w_img, h_img))
            kept_g.append(int(g))
        n_tp += len(kept_b)

        # False positives: sampled near real targets (crowd/duplicate errors),
        # which is harder and more realistic than uniform background boxes.
        n_add = int(rng.poisson(theta.lambd * len(boxes))) if theta.lambd > 0 else 0
        fp_b, fp_g = [], []
        for _ in range(n_add):
            if len(boxes):
                src = boxes[int(rng.integers(0, len(boxes)))]
                sx, sy, sw, sh = src
                ang = rng.uniform(0.0, 2.0 * math.pi)
                rad = rng.uniform(0.5, 1.5)
                nb = np.array([sx + math.cos(ang) * rad * sw,
                               sy + math.sin(ang) * rad * sh,
                               sw * rng.uniform(0.6, 1.4),
                               sh * rng.uniform(0.6, 1.4)], dtype=np.float64)
            else:
                nb = np.array([rng.uniform(0, w_img * 0.8),
                               rng.uniform(0, h_img * 0.8),
                               rng.uniform(20, 80), rng.uniform(40, 160)],
                              dtype=np.float64)
            fp_b.append(_clip_box(nb, w_img, h_img))
            fp_g.append(FP_GID)
        n_fp += n_add

        all_b = kept_b + fp_b
        all_g = np.asarray(kept_g + fp_g, dtype=np.int64)
        if all_b:
            B = np.stack(all_b).astype(np.float32)
        else:
            B = np.zeros((0, 4), dtype=np.float32)
        S = np.concatenate([sc.draw_tp(rng, len(kept_b)),
                            sc.draw_fp(rng, len(fp_b))]).astype(np.float32) \
            if len(all_g) else np.zeros((0,), dtype=np.float32)
        if theta.sigma_loc == 0.0 and theta.beta == 0.0 and theta.lambd == 0.0:
            # Identity condition: exact GT boxes, fixed score, exact ids.
            S = np.ones((len(all_g),), dtype=np.float32)
        out[f] = DetFrame(int(f), B, S, all_g)

    stats = dict(n_frames=len(frames), n_gt=int(n_gt), n_tp=int(n_tp),
                 n_fp=int(n_fp), n_fn=int(n_fn),
                 observed_fn_rate=float(n_fn / max(1, n_gt)),
                 observed_fp_rate=float(n_fp / max(1, n_gt)),
                 theta=theta.as_dict(), seed=int(seed))
    return out, stats


def requires_feature_reextraction(theta: Theta) -> bool:
    """Structural cost rule: only box-geometry changes invalidate cached crops.

    beta deletes boxes, lambda adds boxes, tau changes temporal correlation;
    none of them moves a surviving true-positive box. Therefore the OSNet
    feature of a surviving TP is bit-identical to the one extracted from the
    corresponding GT box and can be reused from a single GT-keyed cache.
    Only sigma_loc != 0 requires re-extraction (plus the newly injected FP
    crops, whose count is ~lambda x n_GT).
    This is an algebraic fact of the injection, NOT an approximation fitted
    to results; phase1_selftest verifies it box by box.
    """
    return float(theta.sigma_loc) > 0.0


def tp_boxes_unchanged(gt: Dict[int, Tuple[np.ndarray, np.ndarray]],
                       det_frames: Dict[int, DetFrame]) -> bool:
    """Verify every surviving TP box is exactly its GT box (=> cache reuse)."""
    for f, (gb, gids) in gt.items():
        if f not in det_frames:
            return False
        m = {int(g): np.asarray(b, dtype=np.float64)
             for b, g in zip(gb, gids)}
        D = det_frames[f]
        for j in range(len(D)):
            g = int(D.src_gid[j])
            if g == FP_GID:
                continue
            if g not in m or not np.array_equal(
                    D.boxes[j].astype(np.float64), m[g]):
                return False
    return True


def grid_conditions(grid: dict) -> List[Theta]:
    """Enumerate the FROZEN condition grid (no result-driven additions).

    Single-factor sweeps: each component at its 6 frozen levels, others at 0,
    tau=1. Interaction 2x2 grids: (sigma_loc x lambda), (beta x lambda),
    (tau x beta), using the frozen interaction levels.
    """
    conds: List[Theta] = []
    for name, levels in (("sigma_loc", grid["sigma_loc"]),
                         ("beta", grid["beta"]),
                         ("lambd", grid["lambd"])):
        for v in levels:
            d = dict(sigma_loc=0.0, beta=0.0, lambd=0.0, tau=1)
            d[name] = float(v)
            conds.append(Theta(**d))
    for L in grid["tau"]:
        conds.append(Theta(0.0, 0.0, 0.0, int(L)))
    for s in grid["interaction"]["sigma_loc"]:
        for l in grid["interaction"]["lambd"]:
            conds.append(Theta(float(s), 0.0, float(l), 1))
    for b in grid["interaction"]["beta"]:
        for l in grid["interaction"]["lambd"]:
            conds.append(Theta(0.0, float(b), float(l), 1))
    for L in grid["interaction"]["tau"]:
        for b in grid["interaction"]["beta"]:
            conds.append(Theta(0.0, float(b), 0.0, int(L)))
    # De-duplicate while preserving order; the zero/zero/zero/1 baseline appears
    # four times across sweeps and must be counted once.
    seen, uniq = set(), []
    for t in conds:
        if t.key() not in seen:
            seen.add(t.key())
            uniq.append(t)
    return uniq
