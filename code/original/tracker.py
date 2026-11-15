#!/usr/bin/env python3
"""Online association tracker implementing the four compared strategies.

Design contract (pre-registered):
  * Input boxes are GT-filtered boxes (oracle-detection, association-only).
  * The tracker NEVER sees GT identities; GT ids are used only afterwards in
    analysis.py for stratification and correctness accounting.
  * Every strategy consumes the identical cached ReID feature matrix.
  * `hist_auc` confidence is estimated on early frames then FROZEN, reproducing
    the historical-confidence / current-frame mismatch of the synthetic study.
  * `frame_calib` confidence uses only the current-frame relative margin.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from exp_config import (IOU_GATE, MAX_AGE, TEMPLATE_PREFIX, TEMPLATE_EMA,
                        ABSTAIN_THETA, SPARSE_TOPK)

BIG = 1e6


def iou_matrix(a, b):
    """a,b: (N,4)/(M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.maximum(0.0, inter_x2 - inter_x1)
    ih = np.maximum(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


@dataclass
class Track:
    tid: int
    box: np.ndarray                 # xywh, last observation
    template: np.ndarray
    last_frame: int
    hits: int = 1
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    n_tpl: int = 1

    def predict(self):
        p = self.box + self.vel
        p[2:] = np.maximum(p[2:], 1.0)
        return np.array([p[0], p[1], p[0] + p[2], p[1] + p[3]], dtype=np.float32)


class Tracker:
    def __init__(self, cfg, seq, feats, rng=None, tau=None):
        self.cfg = cfg
        self.seq = seq
        self.feats = feats
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.kind = cfg["kind"]
        self.conf_mode = cfg.get("confidence", "none")
        self.online = bool(cfg.get("online", False))
        self.K = int(cfg.get("K", 2))
        self.alpha = float(cfg.get("alpha", 0.5))
        self.tau = float(cfg.get("tau", 0.10) if tau is None else tau)
        self.tracks = []
        self.next_tid = 1
        self.hist_auc = None
        self._hist_vals = []
        self._hist_calib_frames = 100
        self.stats = dict(
            n_frames=0, n_dets=0, n_amb=0, n_cand_pairs=0, n_appearance_calls=0,
            n_trigger=0, n_pde=0, n_abstain=0, n_adopt=0, n_disagree=0,
        )
        self.events = []

    # ------------------------------------------------------------ confidence
    def _hist_confidence(self, f, ti, tj, frame_no):
        """Historical AUC proxy: separability of template pairs on early frames, then frozen."""
        if self.hist_auc is None:
            sep = 1.0 - float(np.dot(ti.template, tj.template))
            if frame_no <= self._hist_calib_frames:
                self._hist_vals.append(sep)
                return 0.5  # neutral while calibrating
            m = float(np.mean(self._hist_vals)) if self._hist_vals else 0.5
            self.hist_auc = float(np.clip(1.0 - math.exp(-6.0 * m), 0.05, 0.99))
        return self.hist_auc

    def _frame_confidence(self, s_rel):
        """Current-frame only: confidence from the instantaneous relative margin."""
        return float(1.0 - math.exp(-5.0 * abs(s_rel)))

    # ------------------------------------------------------------ core
    def _relative_score(self, fd, ti, tj):
        """Sparse pairwise-difference arbitration score."""
        v = ti.template - tj.template
        av = np.abs(v)
        k = min(SPARSE_TOPK, len(av))
        # deterministic sparse support with a seeded random component (seed ablation)
        top = np.argpartition(av, -2 * k)[-2 * k:]
        idx = self.rng.choice(top, size=k, replace=False)
        u = (fd - tj.template)[idx]
        w = v[idx]
        denom = float(np.linalg.norm(u) * np.linalg.norm(w) + 1e-9)
        return float(np.dot(u, w) / denom)

    def _appearance_cost(self, ti, fd):
        return float(1.0 - np.dot(ti.template, fd)) * 0.5

    def run(self):
        s = self.seq
        rows = []
        for frame_no in s.frames:
            boxes = s.boxes[frame_no]              # (N,4) xywh
            xyxy = s.xyxy[frame_no]
            off = s.offsets[frame_no]
            F = self.feats[off:off + len(boxes)]
            n = len(boxes)
            self.stats["n_frames"] += 1
            self.stats["n_dets"] += n
            if len(self.tracks) == 0:
                for i in range(n):
                    self.tracks.append(Track(self.next_tid, boxes[i].copy(), F[i].copy(), frame_no))
                    rows.append((frame_no, self.next_tid, *boxes[i].tolist(), 1.0))
                    self.next_tid += 1
                continue

            active = [t for t in self.tracks if frame_no - t.last_frame <= MAX_AGE]
            pred = np.stack([t.predict() for t in active]) if active else np.zeros((0, 4), np.float32)
            iou = iou_matrix(pred, xyxy)
            cand = iou >= IOU_GATE
            m = len(active)
            cost = np.full((m, n), BIG, dtype=np.float64)
            cost[cand] = 1.0 - iou[cand]
            self.stats["n_cand_pairs"] += int(cand.sum())
            self.stats["n_amb"] += int((cand.sum(axis=0) >= 2).sum())

            # ---- S1 / S2: global appearance on (a subset of) candidate pairs
            if self.kind in ("global", "count"):
                if self.kind == "count":
                    det_on = cand.sum(axis=0) >= self.K
                else:
                    det_on = np.ones(n, dtype=bool)
                for d in np.flatnonzero(det_on):
                    for r in np.flatnonzero(cand[:, d]):
                        c = self._appearance_cost(active[r], F[d])
                        cost[r, d] = (1.0 - self.alpha) * (1.0 - iou[r, d]) + self.alpha * c
                        self.stats["n_appearance_calls"] += 1

            # ---- S3: confusion-triggered sparse pairwise-difference arbitration
            if self.kind == "proposed":
                for d in range(n):
                    rs = np.flatnonzero(cand[:, d])
                    if len(rs) < 2:
                        continue
                    cm = cost[rs, d]
                    order = np.argsort(cm)
                    r1, r2 = rs[order[0]], rs[order[1]]
                    margin = float(cm[order[1]] - cm[order[0]])
                    if margin >= self.tau:
                        continue
                    ti, tj = active[r1], active[r2]
                    self.stats["n_trigger"] += 1
                    self.stats["n_pde"] += 1
                    fd = F[d]
                    s_rel = self._relative_score(fd, ti, tj)
                    g_i = float(np.dot(ti.template, fd))
                    g_j = float(np.dot(tj.template, fd))
                    rel_choice = 0 if s_rel > 0 else 1     # 0 -> r1, 1 -> r2
                    glob_choice = 0 if g_i >= g_j else 1
                    disagree = rel_choice != glob_choice
                    if disagree:
                        self.stats["n_disagree"] += 1
                    if self.conf_mode == "hist_auc":
                        conf = self._hist_confidence(fd, ti, tj, self.stats["n_frames"])
                    elif self.conf_mode == "frame_calib":
                        conf = self._frame_confidence(s_rel)
                    else:
                        conf = 1.0
                    adopted = conf >= ABSTAIN_THETA
                    if adopted and disagree:
                        chosen = r1 if rel_choice == 0 else r2
                        other = r2 if rel_choice == 0 else r1
                        cost[chosen, d] = max(0.0, cost[chosen, d] - 0.25)
                        cost[other, d] = min(BIG, cost[other, d] + 0.25)
                        self.stats["n_adopt"] += 1
                    else:
                        self.stats["n_abstain"] += 1
                    self.events.append(dict(
                        frame=int(frame_no), det=int(d),
                        t1=int(ti.tid), t2=int(tj.tid),
                        margin=margin, s_rel=s_rel, g_diff=g_i - g_j,
                        conf=conf, adopted=bool(adopted), disagree=bool(disagree),
                        rel_choice=int(rel_choice), glob_choice=int(glob_choice),
                    ))

            if m and n:
                ri, di = linear_sum_assignment(cost)
                valid = cost[ri, di] < BIG - 1
                ri, di = ri[valid], di[valid]
            else:
                ri, di = np.array([], int), np.array([], int)
            matched_r, matched_d = set(ri.tolist()), set(di.tolist())

            for r, d in zip(ri, di):
                t = active[r]
                new_box = boxes[d]
                if t.hits >= 1:
                    delta = new_box - t.box
                    t.vel = 0.6 * t.vel + 0.4 * delta
                t.box = new_box.copy()
                t.hits += 1
                t.last_frame = frame_no
                if self.online:
                    t.template = TEMPLATE_EMA * t.template + (1 - TEMPLATE_EMA) * F[d]
                    t.template /= (np.linalg.norm(t.template) + 1e-9)
                elif t.n_tpl < TEMPLATE_PREFIX:
                    t.template = (t.template * t.n_tpl + F[d]) / (t.n_tpl + 1)
                    t.template /= (np.linalg.norm(t.template) + 1e-9)
                    t.n_tpl += 1
                rows.append((frame_no, t.tid, *new_box.tolist(), 1.0))

            for d in range(n):
                if d not in matched_d:
                    self.tracks.append(Track(self.next_tid, boxes[d].copy(), F[d].copy(), frame_no))
                    rows.append((frame_no, self.next_tid, *boxes[d].tolist(), 1.0))
                    self.next_tid += 1
            self.tracks = [t for t in self.tracks if frame_no - t.last_frame <= MAX_AGE]

        st = dict(self.stats)
        st["n_tracks"] = self.next_tid - 1
        st["trigger_rate"] = st["n_trigger"] / max(1, st["n_amb"])
        st["cost_ratio"] = st["n_pde"] / max(1, st["n_cand_pairs"])
        st["hist_auc"] = self.hist_auc
        return rows, st, self.events
