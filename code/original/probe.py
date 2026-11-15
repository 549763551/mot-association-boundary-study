#!/usr/bin/env python3
"""Stage-0 probe: verify data format, TrackEval compatibility and ROCm ReID throughput."""
import os
import sys
import time
import glob
import collections
import configparser

import numpy as np

ROOT = "/root/mot_exp"
DATA = os.path.join(ROOT, "data")

BENCHMARKS = {
    "dancetrack": os.path.join(DATA, "dancetrack/val"),
    "mot17": os.path.join(DATA, "mot17/MOT17/train"),
    "mot20": os.path.join(DATA, "mot20/MOT20/train"),
}


def seqinfo(path):
    cp = configparser.ConfigParser()
    cp.read(os.path.join(path, "seqinfo.ini"))
    s = cp["Sequence"]
    return {k: s.get(k) for k in ("name", "seqLength", "imWidth", "imHeight", "frameRate", "imExt")}


def probe_data():
    print("=" * 70)
    print("DATA PROBE")
    for name, root in BENCHMARKS.items():
        seqs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
        if name == "mot17":  # keep only FRCNN detector to avoid triple duplication
            seqs = [s for s in seqs if s.endswith("-FRCNN")]
        print(f"\n--- {name}: {len(seqs)} sequences ---")
        total_rows = total_frames = 0
        conf_counter = collections.Counter()
        cls_counter = collections.Counter()
        id_counts = []
        per_frame = []
        for s in seqs[:3]:
            gt = os.path.join(s, "gt", "gt.txt")
            arr = np.loadtxt(gt, delimiter=",", ndmin=2)
            conf_counter.update(arr[:, 6].astype(int).tolist())
            cls_counter.update(arr[:, 7].astype(int).tolist())
            id_counts.append(len(np.unique(arr[:, 1])))
            fr = collections.Counter(arr[:, 0].astype(int).tolist())
            per_frame.append(np.mean(list(fr.values())))
            info = seqinfo(s)
            total_rows += len(arr)
            total_frames += int(info["seqLength"])
            print(f"  {os.path.basename(s):24s} rows={len(arr):7d} ids={len(np.unique(arr[:,1])):4d} "
                  f"frames={info['seqLength']:>5s} size={info['imWidth']}x{info['imHeight']} "
                  f"avg_dets/frame={np.mean(list(fr.values())):.1f}")
            print(f"    gt head: {arr[0].tolist()}")
            print(f"    det.txt exists: {os.path.exists(os.path.join(s, 'det', 'det.txt'))}")
        all_arr = []
        for s in seqs:
            arr = np.loadtxt(os.path.join(s, "gt", "gt.txt"), delimiter=",", ndmin=2)
            all_arr.append(arr)
            total_rows += len(arr)
        A = np.concatenate(all_arr, axis=0)
        conf_counter.update(A[:, 6].astype(int).tolist())
        cls_counter.update(A[:, 7].astype(int).tolist())
        print(f"  conf(col6) distribution : {dict(conf_counter.most_common(6))}")
        print(f"  class(col7) distribution: {dict(cls_counter.most_common(8))}")
        keep = (A[:, 6] != 0) & (A[:, 7] == 1)
        print(f"  keep(conf!=0 & class==1): {keep.sum()} / {len(A)} = {keep.mean():.3%}")
        print(f"  all sequences: total gt rows (incl. first 3 double counted) ~{total_rows}")
    print("=" * 70)


def probe_model():
    print("MODEL / ROCm PROBE")
    import torch
    sys.path.insert(0, os.path.join(ROOT, "code/deep-person-reid"))
    try:
        from torchreid.models.osnet import osnet_x1_0
        print("  torchreid.models.osnet import: OK")
    except Exception as e:
        print("  torchreid import FAIL:", type(e).__name__, str(e)[:300])
        return
    wpath = os.path.join(ROOT, "weights/osnet_x1_0_msmt17.pth")
    sd = torch.load(wpath, map_location="cpu", weights_only=True)
    print(f"  weight tensors: {len(sd)}, first key: {next(iter(sd))}")
    try:
        model = osnet_x1_0(num_classes=1000, pretrained=False, use_gpu=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("   missing sample:", missing[:5])
        if unexpected:
            print("   unexpected sample:", unexpected[:5])
        # feature extractor: drop classifier
        model.classifier = torch.nn.Identity()
        model.eval()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(dev)
        print(f"  device: {dev}, name: {torch.cuda.get_device_name(0) if dev.type=='cuda' else 'cpu'}")
        x = torch.randn(64, 3, 256, 128, device=dev)
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            reps = 10
            for _ in range(reps):
                f = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
        print(f"  feature dim: {tuple(f.shape)}")
        print(f"  throughput : {reps*64/dt:.1f} crops/s (batch=64, 256x128)")
        est = 650000 / (reps * 64 / dt)
        print(f"  est. 650k crops: {est/60:.1f} min")
    except Exception as e:
        import traceback
        print("  MODEL PROBE FAIL:", type(e).__name__, str(e)[:400])
        traceback.print_exc()
    print("=" * 70)


def probe_trackeval():
    print("TRACKEVAL PROBE")
    sys.path.insert(0, os.path.join(ROOT, "code/TrackEval"))
    try:
        from trackeval.datasets import MotChallenge2DBox
        from trackeval.metrics import HOTA, IDF1, CLEARMOT
        print("  imports OK:", MotChallenge2DBox.__name__, HOTA.__name__, IDF1.__name__, CLEARMOT.__name__)
        import inspect
        src = inspect.getsource(MotChallenge2DBox.get_default_dataset_config)
        print("  default config keys:", [l.strip().split(":")[0].strip("' ") for l in src.splitlines() if "':" in l][:20])
    except Exception as e:
        import traceback
        print("  TRACKEVAL FAIL:", type(e).__name__, str(e)[:300])
        traceback.print_exc()
    print("=" * 70)


if __name__ == "__main__":
    probe_data()
    probe_model()
    probe_trackeval()
