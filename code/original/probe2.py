#!/usr/bin/env python3
"""Stage-0 probe 2: self-contained OSNet load + ROCm throughput + IO speed."""
import os
import sys
import time
import glob
import importlib.util

import numpy as np
import cv2
import torch

ROOT = "/root/mot_exp"
EXP = os.path.join(ROOT, "exp")
MODEL_FILE = os.path.join(EXP, "osnet_model.py")
WEIGHT = os.path.join(ROOT, "weights/osnet_x1_0_msmt17.pth")


def load_osnet_module():
    spec = importlib.util.spec_from_file_location("osnet_model", MODEL_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model():
    om = load_osnet_module()
    model = om.osnet_x1_0(num_classes=1000, pretrained=False, use_gpu=True)
    sd = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  backbone load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("    missing:", missing[:8])
    if unexpected:
        print("    unexpected:", unexpected[:8])
    bad = [k for k in missing if not k.startswith("classifier")]
    print(f"  backbone-only missing (excluding classifier): {len(bad)}")
    assert len(bad) == 0, "backbone weights did not load cleanly"
    model.classifier = torch.nn.Identity()
    model.eval()
    return model


def benchmark():
    print("=" * 70)
    print("ROCm OSNet THROUGHPUT")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    model = build_model().to(dev)
    for bs in (64, 128, 256):
        x = torch.randn(bs, 3, 256, 128, device=dev)
        try:
            with torch.no_grad():
                for _ in range(3):
                    f = model(x)
                torch.cuda.synchronize()
                t0 = time.time()
                reps = 12
                for _ in range(reps):
                    f = model(x)
                torch.cuda.synchronize()
                dt = time.time() - t0
            rate = reps * bs / dt
            print(f"  batch={bs:3d}  dim={tuple(f.shape)}  {rate:8.1f} crops/s  ({dt/reps*1000:.1f} ms/batch)")
            if bs == 256:
                for n in (500000, 1000000, 1500000):
                    print(f"    est {n:,} crops -> {n/rate/60:.1f} min")
        except Exception as e:
            print(f"  batch={bs} FAIL: {type(e).__name__}: {str(e)[:200]}")
    print("=" * 70)


def io_benchmark():
    print("IO / PREPROCESS THROUGHPUT")
    seq = "/root/mot_exp/data/dancetrack/val/dancetrack0005"
    img_paths = sorted(glob.glob(os.path.join(seq, "img1", "*.jpg")))
    frame_to_path = {}
    for p in img_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            frame_to_path[int(stem)] = p
        except ValueError:
            pass
    print(f"  sample image paths: {img_paths[:3]}")
    print(f"  frame number range: {min(frame_to_path)}..{max(frame_to_path)} ({len(frame_to_path)} files)")
    gt = np.loadtxt(os.path.join(seq, "gt", "gt.txt"), delimiter=",", ndmin=2)
    by_frame = {}
    for row in gt:
        by_frame.setdefault(int(row[0]), []).append(row[2:6])
    frames = [f for f in sorted(by_frame) if f in frame_to_path][:120]
    t0 = time.time()
    crops = 0
    for fr in frames:
        img = cv2.imread(frame_to_path[fr])
        if img is None:
            continue
        for (x, y, w, h) in by_frame[fr]:
            x, y, w, h = int(x), int(y), int(w), int(h)
            c = img[max(0, y):y+h, max(0, x):x+w]
            if c.size:
                c = cv2.resize(c, (128, 256))
                crops += 1
    dt = time.time() - t0
    print(f"  {len(frames)} frames / {crops} crops in {dt:.2f}s = {crops/dt:.1f} crops/s (single thread)")
    print(f"  est 1.5M crops single-thread IO+resize: {1500000/(crops/dt)/60:.1f} min")
    print("  => with 8 worker processes: ~{:.1f} min".format(1500000 / (crops / dt * 8) / 60))
    print("=" * 70)


def trackeval_probe():
    print("TRACKEVAL CLASSES")
    sys.path.insert(0, os.path.join(ROOT, "code/TrackEval"))
    from trackeval.metrics import HOTA, CLEAR, Identity
    from trackeval.datasets import MotChallenge2DBox
    from trackeval import Evaluator
    print("  OK:", HOTA.__name__, CLEAR.__name__, Identity.__name__, MotChallenge2DBox.__name__, Evaluator.__name__)
    cfg = MotChallenge2DBox.get_default_dataset_config()
    print("  dataset config:")
    for k in ("GT_FOLDER", "TRACKERS_FOLDER", "SEQMAP_FOLDER", "SEQMAP_FILE", "BENCHMARK", "SPLIT_TO_EVAL", "TRACKERS_TO_EVAL", "CLASSES_TO_EVAL", "INPUT_AS_ZIP", "PRINT_CONFIG"):
        print(f"    {k}: {cfg.get(k)}")
    print("=" * 70)


if __name__ == "__main__":
    benchmark()
    io_benchmark()
    trackeval_probe()
