#!/usr/bin/env python3
"""Extract and cache OSNet ReID features for every GT-filtered box.

One pass per benchmark; all strategies consume exactly the same cached feature
matrix, so no strategy can benefit from different preprocessing or randomness.
"""
import os
import sys
import time
import argparse
import importlib.util
import multiprocessing as mp

import numpy as np
import cv2
import torch

from exp_config import (OSNET_FILE, WEIGHT_MSMT17, WEIGHT_IMAGENET, FEAT,
                        REID_H, REID_W, IMAGENET_MEAN, IMAGENET_STD,
                        EXTRACT_BATCH, EXTRACT_WORKERS, EXTRACT_FP16)
from data_utils import list_sequences, load_sequence, get_feature_path, ensure_dirs

cv2.setNumThreads(0)
_S = None
OUT_SUFFIX = ""


def _out_path(benchmark, name):
    p = get_feature_path(benchmark, name)
    return p if not OUT_SUFFIX else p.replace(".npy", OUT_SUFFIX + ".npy")


def _init_worker(seq):
    global _S
    _S = seq


def _process_frame(f):
    global _S
    boxes = _S.boxes[f]
    path = _S.frame_paths.get(f)
    img = cv2.imread(path) if path else None
    n = len(boxes)
    out = np.zeros((n, 3, REID_H, REID_W), dtype=np.uint8)
    if img is None:
        return f, out
    H, W = img.shape[:2]
    for i, (x, y, w, h) in enumerate(boxes):
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        if x1 <= x0 or y1 <= y0:
            continue
        c = img[y0:y1, x0:x1]
        c = cv2.resize(c, (REID_W, REID_H), interpolation=cv2.INTER_LINEAR)
        c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        out[i] = c.transpose(2, 0, 1)
    return f, out


def build_model(weight_path, fp16):
    spec = importlib.util.spec_from_file_location("osnet_model", OSNET_FILE)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    model = om.osnet_x1_0(num_classes=1000, pretrained=False, use_gpu=True)
    sd = torch.load(weight_path, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("classifier")]
    assert not bad and not unexpected, (bad, unexpected)
    model.classifier = torch.nn.Identity()
    model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    return model, dev, fp16 and dev.type == "cuda"


def infer(model, dev, use_fp16, crops):
    """crops: (N,3,H,W) uint8 -> (N,512) float32 L2-normalized."""
    n = len(crops)
    if n == 0:
        return np.empty((0, 512), dtype=np.float32)
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    parts = []
    with torch.no_grad():
        for s in range(0, n, EXTRACT_BATCH):
            a = torch.from_numpy(np.ascontiguousarray(crops[s:s + EXTRACT_BATCH])).to(dev)
            x = (a.float() / 255.0 - mean) / std
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    f = model(x)
            else:
                f = model(x)
            f = torch.nn.functional.normalize(f.float(), dim=1)
            parts.append(f.cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def _make_pool(seq, workers):
    """Fork pool on Linux (unchanged); thread pool on platforms without fork.

    cv2.imread releases the GIL, so the thread pool still parallelises the IO
    while sharing `_S` without pickling a large SeqData into every worker.
    """
    if sys.platform.startswith("linux"):
        return mp.get_context("fork").Pool(
            workers, initializer=_init_worker, initargs=(seq,))
    from multiprocessing.pool import ThreadPool
    _init_worker(seq)
    return ThreadPool(workers)


def extract_one(benchmark, name, seq_dir, model, dev, use_fp16, overwrite=False):
    out_path = _out_path(benchmark, name)
    if os.path.exists(out_path) and not overwrite:
        return "skip", os.path.getsize(out_path)
    seq = load_sequence(benchmark, name, seq_dir)
    feats = np.zeros((seq.n_boxes, 512), dtype=np.float32)
    t0 = time.time()
    done = 0
    pending = []
    pending_n = 0

    def flush():
        nonlocal pending, pending_n, done
        if not pending:
            return
        cat = np.concatenate([c for _, c in pending], axis=0) if len(pending) > 1 else pending[0][1]
        out = infer(model, dev, use_fp16, cat)
        p = 0
        for off, c in pending:
            feats[off:off + len(c)] = out[p:p + len(c)]
            p += len(c)
        done += len(cat)
        pending = []
        pending_n = 0

    ctx = _make_pool(seq, EXTRACT_WORKERS)
    with ctx as pool:
        for k, (f, crops) in enumerate(pool.imap(_process_frame, seq.frames, chunksize=4)):
            off = seq.offsets[f]
            pending.append((off, crops))
            pending_n += len(crops)
            if pending_n >= EXTRACT_BATCH:
                flush()
            if (k + 1) % 200 == 0:
                el = time.time() - t0
                print(f"    {name}: {k+1}/{len(seq.frames)} frames, {done}/{seq.n_boxes} boxes, "
                      f"{done/el:.0f} boxes/s", flush=True)
        flush()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, feats)
    dt = time.time() - t0
    return "ok", f"{seq.n_boxes} boxes in {dt:.1f}s ({seq.n_boxes/dt:.0f}/s)"


def main():
    global EXTRACT_WORKERS, OUT_SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=list(["dancetrack", "mot17", "mot20"]))
    ap.add_argument("--weight", default="msmt17", choices=("msmt17", "imagenet"))
    ap.add_argument("--fp16", type=int, default=int(EXTRACT_FP16))
    ap.add_argument("--workers", type=int, default=EXTRACT_WORKERS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N sequences per benchmark")
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="only these sequence names (default: all); used by checkpointed runs")
    args = ap.parse_args()
    EXTRACT_WORKERS = args.workers
    wp = WEIGHT_MSMT17 if args.weight == "msmt17" else WEIGHT_IMAGENET
    suffix = "" if args.weight == "msmt17" else "_" + args.weight
    OUT_SUFFIX = suffix
    ensure_dirs()
    model, dev, use_fp16 = build_model(wp, bool(args.fp16))
    print(f"model={args.weight} fp16={use_fp16} dev={dev} workers={EXTRACT_WORKERS}", flush=True)
    for bench in args.benchmarks:
        seqs = list_sequences(bench)
        if args.seqs is not None:
            want = set(args.seqs)
            seqs = [(n, d) for n, d in seqs if n in want]
        if args.limit:
            seqs = seqs[:args.limit]
        print(f"=== {bench}: {len(seqs)} sequences ===", flush=True)
        for i, (name, d) in enumerate(seqs):
            t0 = time.time()
            status, info = extract_one(bench, name, d, model, dev, use_fp16, args.overwrite)
            print(f"  [{i+1}/{len(seqs)}] {name}: {status} {info} ({time.time()-t0:.1f}s)", flush=True)
    print("EXTRACT_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
