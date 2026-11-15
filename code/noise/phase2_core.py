#!/usr/bin/env python3
"""Phase 2 core: six association strategies + TrackEval-equivalent metrics.

Self-contained on purpose (does NOT import full_experiment), so the identical
code runs in the local CPU dry-run and on the ROCm server.

Every constant below is FIXED by frozen_config.json / the pre-registered design;
nothing is fitted to an observed result. The HOTA/AssA implementation is a
line-by-line port of trackeval/metrics/hota.py (verified against the upstream
source shipped in /path/to/mot_exp_assets/code/TrackEval.tar.gz), with the matching
score = global_alignment_score * IoU and AssA___AUC = mean over 19 alphas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# ------------------------------------------------------------------ constants
BIG = 1e6
IOU_GATE = 0.05
MAX_AGE = 30
TEMPLATE_PREFIX = 10
TEMPLATE_EMA = 0.9
FUSION_ALPHA = 0.5
TRIGGER_MARGIN = 0.10
ABSTAIN_THETA = 0.55
SPARSE_TOPK = 64
JACCARD_K = 3
HIGH_SCORE_THRESHOLD = 0.5
ALPHAS = np.arange(0.05, 0.99, 0.05)          # exactly trackeval HOTA.array_labels
STRATEGIES = ("A0", "A1", "A2", "A3", "A4", "A5")
# Main response is pre-declared as A4 - A0 (the measured abstention mechanism).
MAIN_RESPONSE = ("A4", "A0")
# Observable (label-free) features for criterion P. Frozen at <=8 dimensions.
OBSERVABLES = (
    "mean_score", "p10_score", "score_gap", "mean_pair_iou",
    "det_per_frame", "scale_cv", "common_mode_energy", "low_score_frac",
)


# ------------------------------------------------------------------ geometry
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a,b: (N,4)/(M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ix1 = np.maximum(a[:, 0][:, None], b[:, 0][None, :])
    iy1 = np.maximum(a[:, 1][:, None], b[:, 1][None, :])
    ix2 = np.minimum(a[:, 2][:, None], b[:, 2][None, :])
    iy2 = np.minimum(a[:, 3][:, None], b[:, 3][None, :])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    aa = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    ba = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = aa[:, None] + ba[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).copy()
    b[:, 2] += b[:, 0]
    b[:, 3] += b[:, 1]
    return b


# ------------------------------------------------------------------ data
@dataclass
class DetSeq:
    """One injected detection sequence (tracker-visible fields only)."""
    frames: List[int]
    boxes: Dict[int, np.ndarray]        # f -> (N,4) xywh
    xyxy: Dict[int, np.ndarray]         # f -> (N,4) xyxy
    scores: Dict[int, np.ndarray]       # f -> (N,)
    feats: Dict[int, np.ndarray]        # f -> (N,512) L2-normalized
    src_gid: Dict[int, np.ndarray]      # f -> (N,) side channel, NEVER given to tracker


@dataclass
class Track:
    tid: int
    box: np.ndarray
    template: np.ndarray
    last_frame: int
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    n_tpl: int = 1
    hits: int = 1

    def predict(self) -> np.ndarray:
        p = self.box + self.vel
        p[2:] = np.maximum(p[2:], 1.0)
        return np.array([p[0], p[1], p[0] + p[2], p[1] + p[3]], dtype=np.float32)


def _hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cost.size == 0:
        return np.array([], int), np.array([], int)
    r, c = linear_sum_assignment(cost)
    valid = cost[r, c] < BIG - 1
    return r[valid], c[valid]


def _base_cost(iou: np.ndarray, cand: np.ndarray) -> np.ndarray:
    cost = np.full(iou.shape, BIG, dtype=np.float64)
    cost[cand] = 1.0 - iou[cand]
    return cost


def _l2(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(1e-9, np.linalg.norm(a, axis=1, keepdims=True))


def _sim_matrix(active: Sequence[Track], F: np.ndarray, mode: str) -> np.ndarray:
    """(n_det, n_track) cosine similarity, global or set-relative residual."""
    if len(active) == 0 or len(F) == 0:
        return np.zeros((len(F), len(active)), dtype=np.float32)
    T = np.stack([t.template for t in active]).astype(np.float32)
    if mode == "global":
        return (F @ T.T).astype(np.float32)
    mu = F.mean(axis=0, keepdims=True)
    return (_l2(F - mu) @ _l2(T - mu).T).astype(np.float32)


def _cross_agreement(F: np.ndarray, T: np.ndarray, k: int = JACCARD_K) -> np.ndarray:
    """(n_det, n_track) Jaccard of two k-neighbour sets in DET-index space.

    Nd(d) = k nearest dets to det d (F @ F.T); Nr(r) = k nearest dets to
    track r (F @ T.T). Both live in the same index space, so Jaccard is well
    defined. Fully vectorised with a one-hot matrix product.
    """
    n = len(F)
    m = len(T)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)
    kk = min(k, n)
    Nd = np.argsort(-(F @ F.T), axis=1)[:, :kk]              # (n,k)
    Nr = np.argsort(-(F @ T.T), axis=0)[:kk, :].T             # (m,k)
    a = np.zeros((n, n), dtype=np.float32)
    b = np.zeros((m, n), dtype=np.float32)
    np.put_along_axis(a, Nd, 1.0, axis=1)
    np.put_along_axis(b, Nr, 1.0, axis=1)
    inter = a @ b.T                                           # (n,m)
    return (inter / np.maximum(1e-9, 2.0 * kk - inter)).astype(np.float32)


def _relative_score(fd, ti, tj, rng) -> float:
    """Sparse pairwise-difference arbitration score (legacy PDE, A3 only)."""
    v = ti.template - tj.template
    av = np.abs(v)
    k = min(SPARSE_TOPK, len(av))
    top = np.argpartition(av, -2 * k)[-2 * k:]
    idx = rng.choice(top, size=k, replace=False)
    u = (fd - tj.template)[idx]
    w = v[idx]
    return float(np.dot(u, w) / (float(np.linalg.norm(u) * np.linalg.norm(w)) + 1e-9))


# ------------------------------------------------------------------ association
def run_tracker(kind: str, seq: DetSeq, seed: int = 0
                ) -> Tuple[Dict[int, np.ndarray], dict]:
    """Online frame-by-frame association. Returns tids[f] (one per det) + stats."""
    if kind not in STRATEGIES:
        raise ValueError(kind)
    rng = np.random.default_rng(int(seed))
    tracks: List[Track] = []
    next_tid = 1
    tids: Dict[int, np.ndarray] = {}
    st = dict(n_frames=0, n_dets=0, n_amb=0, n_cand_pairs=0,
              n_appearance_calls=0, n_trigger=0, n_abstain=0,
              n_adopt=0, n_disagree=0, n_low_discarded=0)

    for f in seq.frames:
        B = seq.boxes[f]
        X = seq.xyxy[f]
        S = seq.scores[f]
        F = np.asarray(seq.feats[f], dtype=np.float32)
        n = len(B)
        st["n_frames"] += 1
        st["n_dets"] += n

        if not tracks:
            ids = np.arange(next_tid, next_tid + n, dtype=np.int64)
            for i in range(n):
                tracks.append(Track(int(ids[i]), B[i].copy(), F[i].copy(), f))
            next_tid += n
            tids[f] = ids
            continue

        active = [t for t in tracks if f - t.last_frame <= MAX_AGE]
        m = len(active)
        pred = np.stack([t.predict() for t in active]) if m else np.zeros((0, 4), np.float32)
        iou = iou_matrix(pred, X)
        cand = iou >= IOU_GATE
        cost = _base_cost(iou, cand)
        st["n_cand_pairs"] += int(cand.sum())
        st["n_amb"] += int((cand.sum(axis=0) >= 2).sum())
        ids = np.full(n, -1, dtype=np.int64)

        # ---- A1: ByteTrack-style two-stage low-score recovery ---------------
        if kind == "A1":
            high = np.flatnonzero(S >= HIGH_SCORE_THRESHOLD)
            low = np.flatnonzero(S < HIGH_SCORE_THRESHOLD)
            r1, d1 = _hungarian(cost[:, high])
            used_r = set(r1.tolist())
            rem_r = np.array([r for r in range(m) if r not in used_r], dtype=int)
            r2l, d2l = _hungarian(cost[np.ix_(rem_r, low)]) if len(rem_r) and len(low) \
                else (np.array([], int), np.array([], int))
            pairs = [(int(r1[i]), int(high[d1[i]])) for i in range(len(r1))]
            pairs += [(int(rem_r[r2l[i]]), int(low[d2l[i]])) for i in range(len(r2l))]
            matched_h = set(high[d1].tolist())
            new_dets = [int(d) for d in high if d not in matched_h]
            st["n_low_discarded"] += len(low) - len(d2l)
        else:
            # ---- appearance / abstention variants --------------------------
            if kind in ("A2", "A4", "A5"):
                mode = "set_relative" if kind == "A5" else "global"
                sim = _sim_matrix(active, F, mode)
                agree = _cross_agreement(F, np.stack([t.template for t in active])) \
                    if (kind == "A4" and m and n) else None
                for d in range(n):
                    for r in np.flatnonzero(cand[:, d]):
                        if kind == "A4" and agree is not None and \
                                float(agree[d, r]) < ABSTAIN_THETA:
                            st["n_abstain"] += 1
                            continue                       # abstain -> keep pure IoU
                        st["n_appearance_calls"] += 1
                        ap = (1.0 - float(sim[d, r])) * 0.5
                        cost[r, d] = (1.0 - FUSION_ALPHA) * (1.0 - iou[r, d]) \
                            + FUSION_ALPHA * ap
                        if kind == "A4":
                            st["n_adopt"] += 1

            # ---- A3: legacy confusion-triggered PDE arbitration -------------
            if kind == "A3":
                for d in range(n):
                    rs = np.flatnonzero(cand[:, d])
                    if len(rs) < 2:
                        continue
                    cm = cost[rs, d]
                    order = np.argsort(cm)
                    r1, r2 = int(rs[order[0]]), int(rs[order[1]])
                    margin = float(cm[order[1]] - cm[order[0]])
                    if margin >= TRIGGER_MARGIN:
                        continue
                    ti, tj = active[r1], active[r2]
                    st["n_trigger"] += 1
                    fd = F[d]
                    s_rel = _relative_score(fd, ti, tj, rng)
                    g_i = float(np.dot(ti.template, fd))
                    g_j = float(np.dot(tj.template, fd))
                    rel_choice = 0 if s_rel > 0 else 1
                    glob_choice = 0 if g_i >= g_j else 1
                    if rel_choice != glob_choice:
                        st["n_disagree"] += 1
                    conf = 1.0 - math.exp(-5.0 * abs(s_rel))
                    if conf >= ABSTAIN_THETA and rel_choice != glob_choice:
                        chosen = r1 if rel_choice == 0 else r2
                        other = r2 if rel_choice == 0 else r1
                        cost[chosen, d] = max(0.0, cost[chosen, d] - 0.25)
                        cost[other, d] = min(BIG, cost[other, d] + 0.25)
                        st["n_adopt"] += 1
                    else:
                        st["n_abstain"] += 1

            ri, di = _hungarian(cost)
            pairs = [(int(ri[i]), int(di[i])) for i in range(len(ri))]
            matched_d = set(di.tolist())
            new_dets = [d for d in range(n) if d not in matched_d]

        # ---- commit ---------------------------------------------------------
        for r, d in pairs:
            t = active[r]
            nb = B[d]
            if t.hits >= 1:
                t.vel = 0.6 * t.vel + 0.4 * (nb - t.box)
            t.box = nb.copy()
            t.hits += 1
            t.last_frame = f
            if t.n_tpl < TEMPLATE_PREFIX:
                t.template = (t.template * t.n_tpl + F[d]) / (t.n_tpl + 1)
                t.template /= (np.linalg.norm(t.template) + 1e-9)
                t.n_tpl += 1
            ids[d] = t.tid
        for d in new_dets:
            tid = next_tid
            next_tid += 1
            tracks.append(Track(tid, B[d].copy(), F[d].copy(), f))
            ids[d] = tid
        tracks = [t for t in tracks if f - t.last_frame <= MAX_AGE]
        tids[f] = ids

    st["n_tracks"] = next_tid - 1
    st["trigger_rate"] = st["n_trigger"] / max(1, st["n_amb"])
    st["reid_calls"] = st["n_appearance_calls"]
    return tids, st


# ------------------------------------------------------------------ metrics
def hota_metrics(gt: Dict[int, Tuple[np.ndarray, np.ndarray]],
                 det_xyxy: Dict[int, np.ndarray],
                 tids: Dict[int, np.ndarray],
                 with_arrays: bool = False) -> dict:
    """Port of trackeval HOTA.eval_sequence (+ combine as one concatenated seq).

    gt[f]       = (gt_boxes_xyxy (Ngt,4), gt_ids (Ngt,))
    det_xyxy[f] = (Ndet,4)
    tids[f]     = (Ndet,) tracker ids
    Returns AssA/DetA/HOTA (AUC = mean over the 19 alphas), LocA, counts.
    """
    frames = sorted(gt)
    gids_all = sorted({int(g) for f in frames for g in gt[f][1]})
    tids_all = sorted({int(t) for f in frames for t in tids[f] if int(t) >= 0})
    gmap = {g: i for i, g in enumerate(gids_all)}
    tmap = {t: i for i, t in enumerate(tids_all)}
    ng, nt = max(1, len(gmap)), max(1, len(tmap))
    pmc = np.zeros((ng, nt), dtype=np.float64)
    gid_count = np.zeros((ng, 1), dtype=np.float64)
    tid_count = np.zeros((1, nt), dtype=np.float64)
    cache = []
    for f in frames:
        G, gi_raw = gt[f]
        D = det_xyxy[f]
        T = np.asarray(tids[f], dtype=np.int64)
        sim = iou_matrix(G, D)
        gi = np.array([gmap[int(g)] for g in gi_raw], dtype=np.int64)
        tii = np.array([tmap[int(t)] for t in T if int(t) >= 0], dtype=np.int64)
        denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
        sim_iou = np.zeros_like(sim)
        msk = denom > 1e-9
        sim_iou[msk] = sim[msk] / denom[msk]
        np.add.at(pmc, (gi[:, None], tii[None, :]), sim_iou)
        np.add.at(gid_count[:, 0], gi, 1.0)
        np.add.at(tid_count[0], tii, 1.0)
        cache.append((f, sim, gi, tii))

    gal_den = gid_count + tid_count - pmc
    gal = np.where(gal_den > 1e-9, pmc / np.maximum(gal_den, 1e-9), 0.0)

    na = len(ALPHAS)
    tp = np.zeros(na); fn = np.zeros(na); fp = np.zeros(na)
    loca = np.zeros(na)
    matches_counts = [np.zeros((ng, nt), dtype=np.float64) for _ in range(na)]
    for f, sim, gi, tii in cache:
        if len(gi) == 0:
            fp += len(tii)
            continue
        if len(tii) == 0:
            fn += len(gi)
            continue
        score = gal[gi[:, None], tii[None, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        s = sim[rows, cols]
        for a, alpha in enumerate(ALPHAS):
            ok = s >= alpha - 1e-9
            r, c = rows[ok], cols[ok]
            nm = len(r)
            tp[a] += nm
            fn[a] += len(gi) - nm
            fp[a] += len(tii) - nm
            if nm:
                loca[a] += float(s[ok].sum())
                np.add.at(matches_counts[a], (gi[r], tii[c]), 1.0)

    assa = np.zeros(na)
    for a in range(na):
        mc = matches_counts[a]
        ass_a = mc / np.maximum(1.0, gid_count + tid_count - mc)
        assa[a] = float(np.sum(mc * ass_a) / max(1.0, tp[a]))
    deta = tp / np.maximum(1.0, tp + fn + fp)
    hota = np.sqrt(deta * assa)
    loca = np.maximum(1e-10, loca) / np.maximum(1e-10, tp)
    out = dict(AssA=float(np.mean(assa)), DetA=float(np.mean(deta)),
               HOTA=float(np.mean(hota)), LocA=float(np.mean(loca)),
               HOTA_TP=float(np.sum(tp)), HOTA_FN=float(np.sum(fn)),
               HOTA_FP=float(np.sum(fp)))
    if with_arrays:
        out["AssA_array"] = assa.tolist()
        out["DetA_array"] = deta.tolist()
    return out


def idsw_sidechannel(det_frames, tids: Dict[int, np.ndarray]) -> int:
    """ID switches via the sanctioned src_gid side channel.

    prev.update(cur) (NOT prev = cur) so that a target reappearing after an
    occlusion is still counted -- this exact bug was fixed in analyze.py.
    """
    prev: Dict[int, int] = {}
    sw = 0
    for f in sorted(tids):
        D = det_frames[f]
        T = tids[f]
        cur = {}
        for j in range(len(T)):
            g = int(D.src_gid[j])
            if g >= 0 and int(T[j]) >= 0:
                cur[g] = int(T[j])
        for g, t in cur.items():
            if g in prev and prev[g] != t:
                sw += 1
        prev.update(cur)
    return int(sw)


def idf1_sidechannel(det_frames, tids: Dict[int, np.ndarray]) -> float:
    """IDF1 = 2*IDTP / (n_gt_dets + n_tr_det_dets), IDTP by optimal id matching."""
    gset, tset = set(), set()
    for f in sorted(tids):
        D = det_frames[f]
        T = tids[f]
        for j in range(len(T)):
            if int(D.src_gid[j]) >= 0 and int(T[j]) >= 0:
                gset.add(int(D.src_gid[j]))
                tset.add(int(T[j]))
    if not gset or not tset:
        return 0.0
    gl = {g: i for i, g in enumerate(sorted(gset))}
    tl = {t: i for i, t in enumerate(sorted(tset))}
    C = np.zeros((len(gl), len(tl)), dtype=np.float64)
    ngt = ntr = 0
    for f in sorted(tids):
        D = det_frames[f]
        T = tids[f]
        for j in range(len(T)):
            g, t = int(D.src_gid[j]), int(T[j])
            if g >= 0:
                ngt += 1
                if t >= 0:
                    C[gl[g], tl[t]] += 1.0
        ntr += int((T >= 0).sum())
    r, c = linear_sum_assignment(-C)
    idtp = float(C[r, c].sum())
    return float(2.0 * idtp / max(1.0, ngt + ntr))


# ------------------------------------------------------------------ observables
def observable_features(seq: DetSeq, max_pairs: int = 4096) -> dict:
    """<=8 label-free features computed ONLY from detections + their crops.

    These are the inputs of criterion P (RQ2/H2). No GT identity, no GT box,
    and no evaluation label is ever read here.
    """
    sc, gaps, ious, counts, heights, cms, lows = [], [], [], [], [], [], []
    rng = np.random.default_rng(0)
    for f in seq.frames:
        S = np.asarray(seq.scores[f], dtype=np.float64)
        B = np.asarray(seq.boxes[f], dtype=np.float64)
        F = np.asarray(seq.feats[f], dtype=np.float32)
        X = seq.xyxy[f]
        n = len(B)
        counts.append(n)
        if n == 0:
            continue
        sc.extend(S.tolist())
        lows.append(float((S < HIGH_SCORE_THRESHOLD).mean()))
        heights.extend(B[:, 3].tolist())
        if n >= 2:
            s = np.sort(S)[::-1]
            gaps.append(float(s[0] - s[1]))
            M = iou_matrix(X, X)
            iu = M[np.triu_indices(n, k=1)]
            if len(iu) > max_pairs:
                iu = iu[rng.choice(len(iu), max_pairs, replace=False)]
            ious.extend(iu.tolist())
        cms.append(float(np.mean(F, axis=0) @ np.mean(F, axis=0)))
    sc = np.asarray(sc) if sc else np.array([0.0])
    h = np.asarray(heights) if heights else np.array([1.0])
    out = dict(
        mean_score=float(np.mean(sc)),
        p10_score=float(np.quantile(sc, 0.10)),
        score_gap=float(np.mean(gaps)) if gaps else 0.0,
        mean_pair_iou=float(np.mean(ious)) if ious else 0.0,
        det_per_frame=float(np.mean(counts)) if counts else 0.0,
        scale_cv=float(np.std(h) / max(1e-9, np.mean(h))),
        common_mode_energy=float(np.mean(cms)) if cms else 0.0,
        low_score_frac=float(np.mean(lows)) if lows else 0.0,
    )
    assert tuple(out) == OBSERVABLES
    return out
