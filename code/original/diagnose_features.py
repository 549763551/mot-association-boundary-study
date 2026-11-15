#!/usr/bin/env python3
"""Diagnostic: are the pre-conditions of the pairwise-difference hypothesis
actually satisfied by real ReID features on these benchmarks?

Measured (using GT ids for analysis only):
  * pairwise separability AUC: P(cos(same-id) > cos(diff-id))
  * sparse-difference energy: fraction of |f_a - f_b| energy in top-k dims
"""
import os
import json
import numpy as np

from exp_config import BENCHMARKS, SPARSE_TOPK, RESULTS
from data_utils import list_sequences, load_sequence, load_features, save_json


def seq_diag(seq, feats, rng, max_pairs=4000):
    # mean feature per GT identity (clean template)
    byid = {}
    for f in seq.frames:
        off = seq.offsets[f]
        F = feats[off:off + len(seq.boxes[f])]
        for v, gid in zip(F, seq.ids[f]):
            byid.setdefault(int(gid), []).append(v)
    ids = sorted(byid)
    if len(ids) < 2:
        return None
    T = np.stack([np.mean(byid[i], axis=0) for i in ids])
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)

    # Apples-to-apples: BOTH positive and negative pairs use single-frame features.
    samples = {}
    for i in ids:
        v = np.asarray(byid[i])
        k = min(60, len(v))
        samples[i] = v[rng.choice(len(v), k, replace=False)]
    pos = []
    for i in ids:
        v = samples[i]
        if len(v) < 2:
            continue
        a, b = rng.choice(len(v), size=(2, min(40, len(v))), replace=True)
        pos.extend((v[a] * v[b]).sum(axis=1).tolist())
    neg = []
    for _ in range(3000):
        i, j = rng.choice(len(ids), 2, replace=False)
        vi = samples[ids[i]]
        vj = samples[ids[j]]
        neg.append(float(vi[rng.integers(len(vi))] @ vj[rng.integers(len(vj))]))
    if not pos or not neg:
        return None
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    auc = float(np.mean([np.mean(pos > n) for n in rng.choice(neg, min(1000, len(neg)), replace=False)]))
    templ_cos = float(np.mean([float(T[k] @ samples[ids[k]][0]) for k in range(min(len(ids), 200))]))

    # sparse difference energy
    iu = np.triu_indices(len(ids), k=1)
    ratios = []
    for a, b in zip(*iu):
        if len(ratios) >= 500:
            break
        v = T[a] - T[b]
        av = np.sort(np.abs(v))[::-1]
        ratios.append(float(av[:SPARSE_TOPK].sum() / (av.sum() + 1e-9)))
    return dict(auc=auc, pos_mean=float(pos.mean()), neg_mean=float(neg.mean()),
                pos_std=float(pos.std()), neg_std=float(neg.std()),
                templ_cos=templ_cos,
                sparse_energy=float(np.mean(ratios)) if ratios else None,
                n_ids=len(ids), n_neg=len(neg))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    a = ap.parse_args()
    rng = np.random.default_rng(20261115)
    out = {}
    for bench in a.benchmarks:
        seqs = list_sequences(bench)[:6]
        recs = []
        for name, d in seqs:
            s = load_sequence(bench, name, d)
            try:
                f = load_features(bench, name)
            except FileNotFoundError:
                continue
            r = seq_diag(s, f, rng)
            if r:
                r["seq"] = name
                recs.append(r)
        if not recs:
            print(f"{bench}: no valid diagnostics (features may still be extracting)")
            continue
        agg = dict(
            auc=float(np.mean([r["auc"] for r in recs])),
            pos_mean=float(np.mean([r["pos_mean"] for r in recs])),
            neg_mean=float(np.mean([r["neg_mean"] for r in recs])),
            sparse_energy=float(np.mean([r["sparse_energy"] for r in recs])),
            n_sequences=len(recs),
            per_seq={r["seq"]: {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in r.items() if k != "seq"} for r in recs},
        )
        out[bench] = agg
        print(f"DIAG {bench:12s} AUC={agg['auc']:.4f} "
              f"same_id_cos={agg['pos_mean']:.4f} diff_id_cos={agg['neg_mean']:.4f} "
              f"top{SPARSE_TOPK}_energy={agg['sparse_energy']:.4f} "
              f"n_seq={agg['n_sequences']}", flush=True)
    save_json(out, os.path.join(RESULTS, "feature_diagnostics.json"))
    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
