#!/usr/bin/env python3
"""Data layer: sequence discovery, GT loading, TrackEval layout, tracker IO."""
import os
import glob
import json
import shutil
import configparser
from dataclasses import dataclass, field

import numpy as np

from exp_config import (BENCHMARKS, GT_CONF_COL, GT_CLASS_COL, PEDESTRIAN_CLASS,
                        EVALDATA, TRACKS, RESULTS, LOGS, CALIB_FRACTION)


@dataclass
class SeqData:
    name: str
    benchmark: str
    dir: str
    seq_length: int
    frames: list
    boxes: dict = field(default_factory=dict)      # frame -> (N,4) xywh
    xyxy: dict = field(default_factory=dict)       # frame -> (N,4) xyxy
    ids: dict = field(default_factory=dict)        # frame -> (N,) GT ids (analysis only)
    offsets: dict = field(default_factory=dict)    # frame -> global feature-row offset
    frame_paths: dict = field(default_factory=dict)
    n_boxes: int = 0


def list_sequences(benchmark):
    cfg = BENCHMARKS[benchmark]
    dirs = sorted(glob.glob(os.path.join(cfg["root"], cfg["pattern"])))
    return [(os.path.basename(d.rstrip("/")), d) for d in dirs if os.path.isdir(d)]


def read_seqinfo(seq_dir):
    cp = configparser.ConfigParser()
    cp.read(os.path.join(seq_dir, "seqinfo.ini"))
    s = cp["Sequence"]
    return int(s.get("seqLength")), int(s.get("imWidth")), int(s.get("imHeight")), s.get("imExt", ".jpg")


def load_sequence(benchmark, seq_name, seq_dir):
    seq_length, w, h, ext = read_seqinfo(seq_dir)
    arr = np.loadtxt(os.path.join(seq_dir, "gt", "gt.txt"), delimiter=",", ndmin=2)
    keep = (arr[:, GT_CONF_COL] != 0) & (arr[:, GT_CLASS_COL] == PEDESTRIAN_CLASS)
    arr = arr[keep]
    paths = {}
    for p in glob.glob(os.path.join(seq_dir, "img1", "*" + ext)):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            paths[int(stem)] = p
        except ValueError:
            pass
    by_frame = {}
    for row in arr:
        by_frame.setdefault(int(row[0]), []).append(row)
    frames = sorted(by_frame)
    s = SeqData(name=seq_name, benchmark=benchmark, dir=seq_dir,
                seq_length=seq_length, frames=frames)
    off = 0
    for f in frames:
        rows = np.asarray(by_frame[f], dtype=np.float64)
        b = rows[:, 2:6].astype(np.float32)
        s.boxes[f] = b
        xyxy = b.copy()
        xyxy[:, 2] += xyxy[:, 0]
        xyxy[:, 3] += xyxy[:, 1]
        s.xyxy[f] = xyxy
        s.ids[f] = rows[:, 1].astype(np.int64)
        s.offsets[f] = off
        off += len(b)
        s.frame_paths[f] = paths.get(f)
    s.n_boxes = off
    return s


def feature_path(benchmark, seq_name):
    return os.path.join(os.path.dirname(__file__) + "/../features", benchmark, seq_name + ".npy")


def get_feature_path(benchmark, seq_name):
    from exp_config import FEAT
    return os.path.join(FEAT, benchmark, seq_name + ".npy")


def load_features(benchmark, seq_name):
    p = get_feature_path(benchmark, seq_name)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p)


def calibration_split(benchmark):
    """Deterministic disjoint 20% of sequences, used ONLY for nominal tau selection."""
    seqs = [n for n, _ in list_sequences(benchmark)]
    n = max(1, int(round(len(seqs) * CALIB_FRACTION)))
    return set(seqs[:n]), set(seqs[n:])


def _ensure_link(src, dst):
    """Symlink a sequence into the TrackEval GT tree.

    Linux: plain symlink (unchanged behaviour). Windows: symlink with the
    directory flag when Developer Mode is available; otherwise fall back to a
    GT-only copy, which is all MotChallenge2DBox actually reads.
    """
    if os.path.lexists(dst):
        return
    try:
        if os.name == "nt":
            os.symlink(src, dst, target_is_directory=True)
        else:
            os.symlink(src, dst)
        return
    except (OSError, NotImplementedError):
        pass
    os.makedirs(os.path.join(dst, "gt"), exist_ok=True)
    os.makedirs(os.path.join(dst, "img1"), exist_ok=True)
    si = os.path.join(src, "seqinfo.ini")
    if os.path.exists(si):
        shutil.copy2(si, os.path.join(dst, "seqinfo.ini"))
    gt = os.path.join(src, "gt", "gt.txt")
    if os.path.exists(gt):
        shutil.copy2(gt, os.path.join(dst, "gt", "gt.txt"))


def prepare_eval_tree(verbose=True):
    """Build the exact directory layout TrackEval expects, linking raw GT."""
    for bench, cfg in BENCHMARKS.items():
        tag = f"{cfg['te_benchmark']}-{cfg['te_split']}"
        gt_base = os.path.join(EVALDATA, bench, "gt")
        tr_base = os.path.join(EVALDATA, bench, "trackers")
        gt_split = os.path.join(gt_base, tag)
        tr_split = os.path.join(tr_base, tag)
        os.makedirs(gt_split, exist_ok=True)
        os.makedirs(tr_split, exist_ok=True)
        os.makedirs(os.path.join(gt_base, "seqmaps"), exist_ok=True)
        seqs = list_sequences(bench)
        for name, d in seqs:
            link = os.path.join(gt_split, name)
            _ensure_link(os.path.abspath(d), link)
        seqmap = os.path.join(gt_base, "seqmaps", tag + ".txt")
        with open(seqmap, "w") as f:
            f.write("name\n")
            for name, _ in seqs:
                f.write(name + "\n")
        if verbose:
            print(f"  [{bench}] {tag}: {len(seqs)} sequences, gt={gt_split}, trackers={tr_split}")
    return True


def tracker_dir(benchmark, tracker_name):
    cfg = BENCHMARKS[benchmark]
    tag = f"{cfg['te_benchmark']}-{cfg['te_split']}"
    d = os.path.join(EVALDATA, benchmark, "trackers", tag, tracker_name, "data")
    os.makedirs(d, exist_ok=True)
    return d


def write_track_file(benchmark, tracker_name, seq_name, rows):
    """rows: iterable of (frame, track_id, x, y, w, h, conf)."""
    d = tracker_dir(benchmark, tracker_name)
    p = os.path.join(d, seq_name + ".txt")
    with open(p, "w") as f:
        for r in rows:
            f.write("%d,%d,%.3f,%.3f,%.3f,%.3f,%.6f,-1,-1,-1\n" % tuple(r[:7]))
    return p


def ensure_dirs():
    for d in (TRACKS, RESULTS, LOGS, EVALDATA):
        os.makedirs(d, exist_ok=True)
    for b in BENCHMARKS:
        from exp_config import FEAT
        os.makedirs(os.path.join(FEAT, b), exist_ok=True)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set,)):
        return sorted(o)
    raise TypeError(str(type(o)))
