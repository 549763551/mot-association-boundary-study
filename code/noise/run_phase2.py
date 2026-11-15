#!/usr/bin/env python3
"""Phase 2 orchestration for the pre-registered noise-gradient experiment.

Stages (all constants come from the hash-sealed frozen_config.json):
  smoke   : 1 sequence / 1 condition / 1 seed / all strategies. ENGINEERING
            smoke test only -- its numbers are never used in the analysis and
            never used to fit criterion P.
  explore : 5 sequences x first 500 frames per MAIN benchmark (mot17,
            dancetrack). MOT20 is deliberately held out entirely for the
            one-shot leave-one-dataset confirmation (decision D2).
  confirm : all sequences, all frames, one shot, AFTER P is frozen.

Every cell is appended to cells_<stage>.jsonl and flushed immediately, so an
unexpected instance loss costs at most one in-flight cell. A progress.json is
rewritten after each (sequence, condition, seed) group for the backup daemon.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

from noise_injection import (Theta, inject, grid_conditions, EmpiricalScores)
from phase2_core import (DetSeq, xywh_to_xyxy, run_tracker, hota_metrics,
                         idsw_sidechannel, idf1_sidechannel,
                         observable_features, STRATEGIES, MAIN_RESPONSE)
from phase2_features import (load_seq, ImageCache, FeatureBank, build_model,
                             list_benchmark_sequences)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FROZEN = os.path.join(HERE, "frozen_config.json")
# Frozen sampling plan (frozen_config.sampling.explore / decisions.D2)
EXPLORE_SEQS = 5
EXPLORE_FRAMES = 500


def _utc() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _hash(obj: dict) -> str:
    body = {k: v for k, v in obj.items() if k != "integrity"}
    s = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_frozen(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    got, want = _hash(cfg), cfg.get("integrity", {}).get("digest")
    if got != want:
        raise RuntimeError("FROZEN_HASH_MISMATCH computed=%s stored=%s" % (got, want))
    conds = grid_conditions(cfg["grid"])
    if len(conds) != cfg["n_conditions_unique"]:
        raise RuntimeError("condition enumeration mismatch")
    for i, t in enumerate(conds):
        d = cfg["conditions"][i]
        if any(abs(float(t.as_dict()[k]) - float(d[k])) > 1e-9 for k in
               ("sigma_loc", "beta", "lambd")) or int(t.tau) != int(d["tau"]):
            raise RuntimeError("condition %d does not match frozen list" % i)
    return cfg


def load_score_pool(data_root: str, cap: int = 200000) -> list:
    """Detection confidences from ALL official MOT17 TRAIN det.txt files.

    MOT20's det.txt column 7 is a 0/1 validity flag, not a confidence (verified:
    only 2 unique values), so it is deliberately excluded. Each of the 21
    detectors contributes at most cap/n_files uniformly sampled rows, so the
    pool is not dominated by the first few files. This is an INPUT-FIX made
    before any formal cell and is recorded in the stage report; it is never
    changed in response to an observed DeltaAssA.
    """
    files = sorted(glob.glob(os.path.join(data_root, "mot17", "MOT17",
                                          "train", "*", "det", "det.txt")))
    if not files:
        return []
    per = max(1, cap // len(files))
    rng = np.random.default_rng(0)
    vals = []
    for p in files:
        try:
            a = np.loadtxt(p, delimiter=",", ndmin=2)
            if a.shape[1] <= 6:
                continue
            v = np.asarray(a[:, 6], dtype=np.float64)
            v = np.clip(v[np.isfinite(v)], 0.0, 1.0)
            if len(v) > per:
                v = v[rng.choice(len(v), per, replace=False)]
            vals.extend(v.tolist())
        except Exception:
            pass
    return vals[:cap]


def make_detseq(det_frames, feats) -> DetSeq:
    frames = sorted(det_frames)
    return DetSeq(
        frames=frames,
        boxes={f: det_frames[f].boxes for f in frames},
        xyxy={f: xywh_to_xyxy(det_frames[f].boxes) for f in frames},
        scores={f: det_frames[f].scores for f in frames},
        feats=feats,
        src_gid={f: det_frames[f].src_gid for f in frames})


def evaluate_strategy(seqdata: dict, det_frames, ds: DetSeq,
                      strategy: str, seed: int) -> dict:
    tids, st = run_tracker(strategy, ds, seed=seed)
    keep = {f: (tids[f] >= 0) for f in ds.frames}
    det_xyxy = {f: ds.xyxy[f][keep[f]] for f in ds.frames}
    tids_e = {f: tids[f][keep[f]] for f in ds.frames}
    m = hota_metrics(seqdata["gt_eval"], det_xyxy, tids_e)
    m["IDSW"] = idsw_sidechannel(det_frames, tids)
    m["IDF1"] = idf1_sidechannel(det_frames, tids)
    m["n_output_dets"] = int(sum(int(k.sum()) for k in keep.values()))
    m["n_tracks"] = int(st["n_tracks"])
    for k in ("reid_calls", "trigger_rate", "n_abstain", "n_adopt",
              "n_trigger", "n_amb", "n_low_discarded", "n_cand_pairs"):
        m[k] = float(st[k])
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("smoke", "explore", "confirm"),
                    default="explore")
    ap.add_argument("--frozen", default=DEFAULT_FROZEN)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", choices=("gpu", "toy"), default="gpu")
    ap.add_argument("--osnet-file", default=None)
    ap.add_argument("--weight", default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--img-cache-limit", type=int, default=0)
    ap.add_argument("--benchmarks", nargs="+", default=None)
    ap.add_argument("--n-seqs", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--conds", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    root = os.environ.get("MOT_EXP_ROOT", "/root/mot_exp")
    data_root = a.data_root or os.path.join(root, "data")
    out = a.out or os.path.join(root, "exp_noise", "results")
    os.makedirs(out, exist_ok=True)

    cfg = load_frozen(a.frozen)
    conds = grid_conditions(cfg["grid"])
    seeds = a.seeds or list(cfg["sampling"]["seeds"])
    strategies = a.strategies or list(STRATEGIES)
    for s in strategies:
        if s not in STRATEGIES:
            raise ValueError("unknown strategy " + s)

    if a.conds is not None:
        conds = [conds[i] for i in a.conds]

    if a.stage == "smoke":
        benchmarks = a.benchmarks or ["mot17"]
        n_seqs, max_frames = 1, (a.max_frames or 200)
        if a.conds is None:
            conds = [conds[0]]
        if a.seeds is None:
            seeds = seeds[:1]
    elif a.stage == "explore":
        benchmarks = a.benchmarks or list(cfg["decisions"]["D2_datasets"]["main"])
        n_seqs = a.n_seqs if a.n_seqs is not None else EXPLORE_SEQS
        max_frames = a.max_frames if a.max_frames is not None else EXPLORE_FRAMES
    else:
        benchmarks = a.benchmarks or ["mot17", "dancetrack", "mot20"]
        n_seqs = a.n_seqs if a.n_seqs is not None else 0
        max_frames = a.max_frames if a.max_frames is not None else 0

    # Per-sequence shards keep every file well under the backup daemon's
    # 2 MB mirror limit; a lost instance then costs at most one in-flight cell.
    cells_dir = os.path.join(out, "cells_%s" % a.stage)
    os.makedirs(cells_dir, exist_ok=True)
    progress_path = os.path.join(out, "progress_%s.json" % a.stage)
    done = set()
    if a.resume:
        for p in sorted(glob.glob(os.path.join(cells_dir, "*.jsonl"))):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        done.add("%s|%s|%d|%d" % (r["benchmark"], r["seq"],
                                                  r["cond"], r["seed"]))
                    except Exception:
                        pass

    print("=== Phase 2 / %s ===" % a.stage, flush=True)
    print("  frozen sha256 :", cfg["integrity"]["digest"], flush=True)
    print("  conditions    :", len(conds), " seeds:", seeds, flush=True)
    print("  strategies    :", strategies, flush=True)
    print("  benchmarks    :", benchmarks, " seqs/bench:", n_seqs or "ALL",
          " frames:", max_frames or "ALL", flush=True)
    print("  backend       :", a.backend, " out:", out, flush=True)
    print("  resume cells  :", len(done), flush=True)

    model = dev = None
    if a.backend == "gpu":
        osnet = a.osnet_file or os.path.join(root, "exp", "osnet_model.py")
        weight = a.weight or os.path.join(root, "weights", "osnet_x1_0_msmt17.pth")
        if not os.path.exists(osnet) or not os.path.exists(weight):
            raise FileNotFoundError("osnet=%s weight=%s" % (osnet, weight))
        t0 = time.time()
        model, dev = build_model(osnet, weight)
        print("  model loaded in %.1fs dev=%s" % (time.time() - t0, dev), flush=True)

    pool = load_score_pool(data_root)
    scores = EmpiricalScores(pool)
    pv = np.asarray(pool, dtype=np.float64) if pool else np.array([0.0])
    pool_stats = dict(n=len(pool), mean=float(pv.mean()), std=float(pv.std()),
                      p10=float(np.quantile(pv, .10)),
                      p50=float(np.quantile(pv, .50)),
                      p90=float(np.quantile(pv, .90)),
                      n_unique=int(len(np.unique(pv))), mode=scores.mode,
                      source="all MOT17 TRAIN det.txt, uniform per-file sample, "
                             "clipped to [0,1]; fixed before any formal cell")
    print("  score pool    : %d values, mode=%s std=%.4f unique=%d" %
          (len(pool), scores.mode, pool_stats["std"], pool_stats["n_unique"]),
          flush=True)
    all_seqs = list_benchmark_sequences(data_root)
    t_start = time.time()
    n_cells = 0
    n_group = 0
    try:
        for bench in benchmarks:
            seqs = all_seqs.get(bench, [])
            if n_seqs:
                seqs = seqs[:n_seqs]
            if not seqs:
                print("  WARN no sequences for %s" % bench, flush=True)
                continue
            for si, (name, sdir) in enumerate(seqs):
                t_seq = time.time()
                sd = load_seq(sdir, max_frames)
                images = ImageCache(sd, limit=a.img_cache_limit)
                bank = FeatureBank(sd, images, model, dev, a.batch, a.backend)
                sh = int(hashlib.sha1(("%s/%s" % (bench, name)).encode())
                         .hexdigest()[:8], 16) % 100000
                cell_file = os.path.join(cells_dir, "%s__%s.jsonl" % (bench, name))
                fh = open(cell_file,
                          "a" if (a.resume and os.path.exists(cell_file)) else "w",
                          encoding="utf-8")
                print("  [%s] %s: %d frames (of %d), GT-cache done, "
                      "extracted=%d" % (bench, name, len(sd["frames"]),
                                        sd["n_frames_total"], bank.n_extracted),
                      flush=True)
                for ci, theta in enumerate(conds):
                    for seed in seeds:
                        gkey = "%s|%s|%d|%d" % (bench, name, ci, seed)
                        if gkey in done:
                            continue
                        composed = int(seed) * 1000003 + ci * 10007 + sh
                        det_frames, inj = inject(sd["gt_for_inject"], theta,
                                                 composed, sd["img_size"], scores)
                        feats = bank.det_features_for(det_frames, theta)
                        ds = make_detseq(det_frames, feats)
                        obs = observable_features(ds)
                        for strat in strategies:
                            m = evaluate_strategy(sd, det_frames, ds, strat,
                                                  seed=int(seed))
                            row = dict(
                                stage=a.stage, benchmark=bench, seq=name,
                                cond=ci, theta=theta.as_dict(), seed=int(seed),
                                composed_seed=composed, strategy=strat,
                                n_frames=len(sd["frames"]),
                                inj_n_gt=inj["n_gt"], inj_n_tp=inj["n_tp"],
                                inj_n_fp=inj["n_fp"], inj_n_fn=inj["n_fn"],
                                AssA=m["AssA"], DetA=m["DetA"], HOTA=m["HOTA"],
                                IDSW=m["IDSW"], IDF1=m["IDF1"],
                                HOTA_TP=m["HOTA_TP"], HOTA_FN=m["HOTA_FN"],
                                HOTA_FP=m["HOTA_FP"],
                                reid_calls=m["reid_calls"],
                                trigger_rate=m["trigger_rate"],
                                n_amb=m["n_amb"],
                                n_cand_pairs=m["n_cand_pairs"],
                                n_abstain=m["n_abstain"],
                                n_adopt=m["n_adopt"],
                                n_output_dets=m["n_output_dets"],
                                n_tracks=m["n_tracks"],
                                observables=obs,
                                frozen_sha256=cfg["integrity"]["digest"],
                                utc=_utc())
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                            fh.flush()
                            n_cells += 1
                        n_group += 1
                        with open(progress_path, "w", encoding="utf-8") as pf:
                            json.dump(dict(stage=a.stage, utc=_utc(),
                                           last_group=gkey, groups=n_group,
                                           cells=n_cells,
                                           elapsed_s=round(time.time() - t_start, 1),
                                           frozen_sha256=cfg["integrity"]["digest"],
                                           bank_extracted=bank.n_extracted,
                                           bank_reused=bank.n_reused), pf,
                                      indent=2)
                        if n_group % 20 == 0:
                            el = time.time() - t_start
                            print("    groups=%d cells=%d elapsed=%.0fs "
                                  "(%.1f cells/s)" %
                                  (n_group, n_cells, el, n_cells / max(1, el)),
                                  flush=True)
                fh.close()
                del bank, images
                print("  [%s] %s done in %.0fs (total cells=%d)" %
                      (bench, name, time.time() - t_seq, n_cells), flush=True)
    finally:
        if "fh" in locals() and not fh.closed:
            fh.close()

    report = dict(stage=a.stage, utc=_utc(), status="PHASE2_STAGE_DONE",
                  frozen_sha256=cfg["integrity"]["digest"],
                  n_conditions=len(conds), seeds=seeds,
                  strategies=strategies, benchmarks=benchmarks,
                  n_seqs_per_benchmark=n_seqs or "ALL",
                  max_frames=max_frames or "ALL",
                  groups=n_group, cells=n_cells,
                  elapsed_s=round(time.time() - t_start, 1),
                  main_response=list(MAIN_RESPONSE),
                  score_pool=pool_stats,
                  note="smoke stage is engineering-only; never used to fit P; "
                       "AssA uses official HOTA alignment-weighted matching, so "
                       "the TP set can legitimately vary with the tracker id "
                       "assignment even when detection boxes are identical")
    with open(os.path.join(out, "run_%s.json" % a.stage), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("=== PHASE2_STAGE_DONE cells=%d elapsed=%.0fs ===" %
          (n_cells, time.time() - t_start), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
