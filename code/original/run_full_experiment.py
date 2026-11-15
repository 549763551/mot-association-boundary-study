#!/usr/bin/env python3
"""Main orchestrator for the full real-benchmark experiment.

Stages:
  track    - run every strategy/config over every sequence, write MOT-format results
  eval     - official TrackEval (HOTA/CLEAR/Identity) for all trackers
  analyze  - stratification, budget curves, bootstrap CIs, pre-registered decisions
  all      - the three above in order
"""
import os
import sys
import time
import json
import argparse

import numpy as np

from exp_config import (STRATEGIES, TAU_GRID, SEEDS, RESULTS, BENCHMARKS, ALPHA_GRID)
from data_utils import (list_sequences, load_sequence, load_features,
                        write_track_file, prepare_eval_tree, ensure_dirs, save_json)
from tracker import Tracker


def tau_label(i):
    return f"tau_{i:02d}"


def build_run_list():
    runs = []
    for name, cfg in STRATEGIES.items():
        runs.append((name, dict(cfg), 0))
    for i, tau in enumerate(TAU_GRID):
        cfg = dict(STRATEGIES["S3_frame_calib"])
        cfg["tau"] = tau
        runs.append((tau_label(i), cfg, 0))
    for seed in SEEDS:
        cfg = dict(STRATEGIES["S3_frame_calib"])
        runs.append((f"seed_{seed}", cfg, seed))
    for i, alpha in enumerate(ALPHA_GRID):
        cfg = dict(STRATEGIES["S1_global"])
        cfg["alpha"] = alpha
        runs.append((f"alpha_{i:02d}", cfg, 0))
    return runs


def stage_track(benchmarks, run_list, limit=0):
    ensure_dirs()
    prepare_eval_tree(verbose=True)
    all_stats = {}
    t_all = time.time()
    for bench in benchmarks:
        seqs = list_sequences(bench)
        if limit:
            seqs = seqs[:limit]
        print(f"=== TRACK {bench}: {len(seqs)} seqs x {len(run_list)} runs ===", flush=True)
        for si, (name, d) in enumerate(seqs):
            t0 = time.time()
            seq = load_sequence(bench, name, d)
            feats = load_features(bench, name)
            assert feats.shape[0] == seq.n_boxes, (name, feats.shape, seq.n_boxes)
            for run_name, cfg, seed in run_list:
                tr = Tracker(cfg, seq, feats, rng=np.random.default_rng(seed))
                rows, st, _ = tr.run()
                write_track_file(bench, run_name, name, rows)
                all_stats.setdefault(f"{bench}/{run_name}", {})[name] = st
            el = time.time() - t0
            print(f"  [{si+1}/{len(seqs)}] {name}: {seq.n_boxes} boxes, {el:.1f}s "
                  f"({len(run_list)} runs)", flush=True)
    p = os.path.join(RESULTS, "run_stats.json")
    if os.path.exists(p):
        try:
            prev = json.load(open(p))
            for k, v in all_stats.items():
                prev.setdefault(k, {}).update(v)
            all_stats = prev
        except Exception as e:
            print("  warn: could not merge previous stats:", e, flush=True)
    save_json(all_stats, p)
    print(f"TRACK_DONE total={time.time()-t_all:.1f}s", flush=True)
    return all_stats


def stage_eval(benchmarks, run_list):
    from evaluate import evaluate_all
    names = [r[0] for r in run_list]
    return evaluate_all(names, benchmarks)


def stage_analyze(benchmarks, run_list):
    from analyze import analyze_all
    return analyze_all(benchmarks, [r[0] for r in run_list])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("track", "eval", "analyze", "all"))
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    runs = build_run_list()
    print(f"runs={len(runs)} benchmarks={a.benchmarks} stage={a.stage}", flush=True)
    if a.stage in ("track", "all"):
        stage_track(a.benchmarks, runs, a.limit)
    if a.stage in ("eval", "all"):
        stage_eval(a.benchmarks, runs)
    if a.stage in ("analyze", "all"):
        stage_analyze(a.benchmarks, runs)


if __name__ == "__main__":
    main()
