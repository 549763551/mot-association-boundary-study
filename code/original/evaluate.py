#!/usr/bin/env python3
"""TrackEval wrapper: official HOTA / CLEAR / Identity metrics."""
import os
import sys
import glob
import json
import copy

import numpy as np
import pandas as pd

# TrackEval (upstream) still uses the removed NumPy 1.x scalar aliases.
# Restore them as their exact builtin equivalents; no numeric semantics change.
for _n, _t in (("float", float), ("int", int), ("bool", bool),
               ("object", object), ("str", str)):
    if not hasattr(np, _n):
        setattr(np, _n, _t)

from exp_config import BENCHMARKS, EVALDATA, RESULTS
from data_utils import prepare_eval_tree, save_json

_TRACKEVAL_DIR = os.environ.get(
    "MOT_TRACKEVAL_DIR",
    os.path.join(os.environ.get("MOT_EXP_ROOT", "/root/mot_exp"), "code", "TrackEval"))
sys.path.insert(0, _TRACKEVAL_DIR)


def _build_config(bench, trackers):
    cfg = BENCHMARKS[bench]
    dataset_config = {
        "GT_FOLDER": os.path.join(EVALDATA, bench, "gt"),
        "TRACKERS_FOLDER": os.path.join(EVALDATA, bench, "trackers"),
        "OUTPUT_FOLDER": None,
        "TRACKERS_TO_EVAL": list(trackers),
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": cfg["te_benchmark"],
        "SPLIT_TO_EVAL": cfg["te_split"],
        "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": False,
        "DO_PREPROC": True,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": None,
        "SEQMAP_FOLDER": None,
        "SEQMAP_FILE": None,
        "SEQ_INFO": None,
    }
    eval_config = {
        "USE_PARALLEL": False,
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
        "BREAK_ON_ERROR": True,
    }
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
    return dataset_config, eval_config, metrics_config


def _detailed_csv(tracker_dir_path):
    p = os.path.join(tracker_dir_path, "pedestrian_detailed.csv")
    if os.path.exists(p):
        return p
    cands = glob.glob(os.path.join(tracker_dir_path, "**", "pedestrian_detailed.csv"), recursive=True)
    return cands[0] if cands else None


def _read_summary(tracker_dir_path):
    """TrackEval writes its official aggregate as the COMBINED row of detailed.csv."""
    p = _detailed_csv(tracker_dir_path)
    if not p:
        return {}
    df = pd.read_csv(p)
    row = df[df["seq"].astype(str).str.upper() == "COMBINED"]
    if len(row) == 0:
        return {}
    out = {}
    for k, v in row.iloc[0].items():
        if k == "seq":
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    for alias, src in (("HOTA", "HOTA___AUC"), ("AssA", "AssA___AUC"),
                       ("DetA", "DetA___AUC"), ("AssRe", "AssRe___AUC"),
                       ("AssPr", "AssPr___AUC"), ("LocA", "LocA___AUC")):
        if src in out:
            out[alias] = out[src]
    return out


def _read_detailed(tracker_dir_path):
    p = _detailed_csv(tracker_dir_path)
    if not p:
        return {}
    df = pd.read_csv(p)
    out = {}
    for _, row in df.iterrows():
        seq = str(row["seq"])
        if seq.upper() == "COMBINED":
            continue
        out[seq] = {k: float(v) for k, v in row.items() if k != "seq" and pd.notna(v)}
    return out


def evaluate_all(tracker_names, benchmarks=None):
    """tracker_names: list; each must already have result files for all benchmarks."""
    import trackeval
    prepare_eval_tree(verbose=False)
    benchmarks = benchmarks or list(BENCHMARKS)
    results = {}
    for bench in benchmarks:
        tag = f"{BENCHMARKS[bench]['te_benchmark']}-{BENCHMARKS[bench]['te_split']}"
        base = os.path.join(EVALDATA, bench, "trackers", tag)
        available = [t for t in tracker_names if os.path.isdir(os.path.join(base, t, "data"))]
        if not available:
            print(f"  [{bench}] no trackers found, skip")
            continue
        dcfg, ecfg, mcfg = _build_config(bench, available)
        print(f"  [{bench}] evaluating {len(available)} trackers: {available[:4]}{'...' if len(available)>4 else ''}", flush=True)
        evaluator = trackeval.Evaluator(ecfg)
        dataset = trackeval.datasets.MotChallenge2DBox(dcfg)
        metrics = []
        for m in (trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity):
            if m.get_name() in mcfg["METRICS"]:
                metrics.append(m(mcfg))
        evaluator.evaluate([dataset], metrics)
        for t in available:
            tdir = os.path.join(base, t)
            summary = _read_summary(tdir)
            detailed = _read_detailed(tdir)
            results[f"{bench}/{t}"] = {"summary": summary, "per_seq": detailed}
            hota = summary.get("HOTA", float("nan"))
            idf1 = summary.get("IDF1", float("nan"))
            idsw = summary.get("IDSW", float("nan"))
            print(f"    {t:28s} HOTA={hota:.4f} IDF1={idf1:.4f} IDSW={idsw:.0f}", flush=True)
    # Merge with any previously evaluated benchmarks so that per-benchmark
    # staged runs (needed for checkpointing on an unreliable instance) do not
    # overwrite each other. Fresh full runs are unaffected.
    p = os.path.join(RESULTS, "trackeval_results.json")
    if os.path.exists(p):
        try:
            prev = json.load(open(p))
            for k, v in results.items():
                prev[k] = v
            results = prev
        except Exception as e:
            print("  warn: could not merge previous eval results:", e, flush=True)
    save_json(results, p)
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=None)
    a = ap.parse_args()
    evaluate_all(a.trackers, a.benchmarks)
