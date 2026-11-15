#!/usr/bin/env python3
"""Feature bank for Phase 2.

Geometry-keyed reuse (frozen_config.feature_cache.rule = geometry_keyed_v1):
  * GT-keyed cache is extracted ONCE per sequence from the GT boxes.
  * When sigma_loc == 0, every surviving TP box is bit-identical to its GT box
    (phase1_selftest verifies this), so its feature is reused from that cache.
  * Injected FP crops and all boxes when sigma_loc > 0 are extracted on demand.
  * A deterministic toy backend (CPU, no images, no weights) drives the local
    dry-run so the full pipeline can be validated without GPU compute or large transfers.

The backend is self-contained: no import of full_experiment, so it works both
locally and on the ROCm instance.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import configparser
from typing import Dict, Optional, Tuple

import numpy as np

from noise_injection import FP_GID, requires_feature_reextraction

REID_H, REID_W = 256, 128
FEAT_DIM = 512
GT_CONF_COL, GT_CLASS_COL, PEDESTRIAN_CLASS = 6, 7, 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ sequence
def load_seq(seq_dir: str, max_frames: int = 0) -> dict:
    """Load a MOT-format sequence with the exact TrackEval pedestrian filter."""
    cp = configparser.ConfigParser()
    cp.read(os.path.join(seq_dir, "seqinfo.ini"))
    s = cp["Sequence"]
    w, h = int(s.get("imWidth")), int(s.get("imHeight"))
    ext = s.get("imExt", ".jpg")
    arr = np.loadtxt(os.path.join(seq_dir, "gt", "gt.txt"), delimiter=",", ndmin=2)
    keep = (arr[:, GT_CONF_COL] != 0) & (arr[:, GT_CLASS_COL] == PEDESTRIAN_CLASS)
    arr = arr[keep]
    by_frame: Dict[int, list] = {}
    for row in arr:
        by_frame.setdefault(int(row[0]), []).append(row)
    frames = sorted(by_frame)
    n_total = len(frames)
    if max_frames:
        frames = frames[:max_frames]
    paths = {}
    for p in glob.glob(os.path.join(seq_dir, "img1", "*" + ext)):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            paths[int(stem)] = p
        except ValueError:
            pass
    gt_xywh, gt_xyxy, gids = {}, {}, {}
    for f in frames:
        rows = np.asarray(by_frame[f], dtype=np.float64)
        b = rows[:, 2:6].astype(np.float64)
        gt_xywh[f] = b
        x = b.copy()
        x[:, 2] += x[:, 0]
        x[:, 3] += x[:, 1]
        gt_xyxy[f] = x
        gids[f] = rows[:, 1].astype(np.int64)
    return dict(dir=seq_dir, frames=frames, n_frames_total=n_total,
                gt_xywh=gt_xywh, gt_xyxy=gt_xyxy, gt_gids=gids,
                gt_for_inject={f: (gt_xywh[f], gids[f]) for f in frames},
                gt_eval={f: (gt_xyxy[f], gids[f]) for f in frames},
                frame_paths=paths, img_size=(float(w), float(h)))


# ------------------------------------------------------------------ images
class ImageCache:
    """Cache the frames of ONE sequence; released before the next sequence."""

    def __init__(self, seq: dict, limit: int = 0):
        self.paths = seq["frame_paths"]
        self.limit = int(limit)
        self.cache: Dict[int, Optional[np.ndarray]] = {}
        self.n_read = 0

    def get(self, f: int):
        if f in self.cache:
            return self.cache[f]
        import cv2
        p = self.paths.get(f)
        img = cv2.imread(p) if p else None
        self.n_read += 1
        if self.limit:
            if len(self.cache) >= self.limit:
                self.cache.pop(next(iter(self.cache)))
        self.cache[f] = img
        return img


def make_crops(img, boxes: np.ndarray) -> np.ndarray:
    import cv2
    n = len(boxes)
    out = np.zeros((n, 3, REID_H, REID_W), dtype=np.uint8)
    if img is None:
        return out
    H, W = img.shape[:2]
    for i, (x, y, w, h) in enumerate(boxes):
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        if x1 <= x0 or y1 <= y0:
            continue
        c = cv2.resize(img[y0:y1, x0:x1], (REID_W, REID_H),
                       interpolation=cv2.INTER_LINEAR)
        out[i] = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    return out


# ------------------------------------------------------------------ OSNet
def build_model(osnet_file: str, weight_path: str):
    import torch
    spec = importlib.util.spec_from_file_location("osnet_model", osnet_file)
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
    return model.to(dev), dev


def infer(model, dev, crops: np.ndarray, batch: int = 256) -> np.ndarray:
    import torch
    if len(crops) == 0:
        return np.empty((0, FEAT_DIM), dtype=np.float32)
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    parts = []
    with torch.no_grad():
        for s in range(0, len(crops), batch):
            a = torch.from_numpy(np.ascontiguousarray(crops[s:s + batch])).to(dev)
            f = model((a.float() / 255.0 - mean) / std)
            parts.append(torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


# ------------------------------------------------------------------ bank
class FeatureBank:
    """GT-keyed cache plus on-demand extraction for injected crops."""

    def __init__(self, seq: dict, images: ImageCache,
                 model=None, dev=None, batch: int = 256, backend: str = "gpu"):
        self.seq = seq
        self.images = images
        self.model = model
        self.dev = dev
        self.batch = batch
        self.backend = backend
        self.n_extracted = 0
        self.n_reused = 0
        self.gt_feat: Dict[int, np.ndarray] = {}
        self.gid_row: Dict[int, dict] = {}
        self._build_gt_cache()

    def _raw(self, boxes_by_frame: Dict[int, np.ndarray]):
        """Vectorised crop+infer over a dict of frame -> boxes."""
        frames = sorted(boxes_by_frame)
        offsets, all_crops = {}, []
        p = 0
        for f in frames:
            b = boxes_by_frame[f]
            offsets[f] = p
            if self.backend == "toy":
                pass
            else:
                all_crops.append(make_crops(self.images.get(f), b))
            p += len(b)
        if self.backend == "toy":
            return {}, offsets
        crops = np.concatenate(all_crops, axis=0) if all_crops \
            else np.zeros((0, 3, REID_H, REID_W), np.uint8)
        feats = infer(self.model, self.dev, crops, self.batch) if len(crops) else \
            np.zeros((0, FEAT_DIM), np.float32)
        self.n_extracted += len(crops)
        return {"feats": feats}, offsets

    def _toy(self, det_frames) -> Dict[int, np.ndarray]:
        """Deterministic identity-consistent pseudo features (local dry-run)."""
        base: Dict[int, np.ndarray] = {}
        out = {}
        for f, D in det_frames.items():
            F = np.zeros((len(D), FEAT_DIM), dtype=np.float32)
            for j, g in enumerate(D.src_gid):
                g = int(g)
                if g == FP_GID:
                    v = np.random.default_rng(900000 + f * 101 + j).normal(size=FEAT_DIM)
                else:
                    if g not in base:
                        base[g] = np.random.default_rng(g + 1).normal(size=FEAT_DIM)
                    v = base[g] + 0.15 * np.random.default_rng(
                        g * 100003 + f).normal(size=FEAT_DIM)
                F[j] = v / max(1e-9, np.linalg.norm(v))
            out[f] = F
        return out

    def _build_gt_cache(self):
        if self.backend == "toy":
            # GT cache is not needed for the toy backend; det_features handles it.
            return
        boxes = {f: np.asarray(self.seq["gt_xywh"][f], dtype=np.float64)
                 for f in self.seq["frames"]}
        res, offsets = self._raw(boxes)
        feats = res["feats"]
        for f in self.seq["frames"]:
            n = len(boxes[f])
            o = offsets[f]
            self.gt_feat[f] = feats[o:o + n]
            self.gid_row[f] = {int(g): i for i, g in
                               enumerate(self.seq["gt_gids"][f])}

    def det_features_for(self, det_frames, theta) -> Dict[int, np.ndarray]:
        """Return per-frame (N,512) features for an injected detection sequence."""
        if self.backend == "toy":
            return self._toy(det_frames)
        reuse = not requires_feature_reextraction(theta)
        out: Dict[int, np.ndarray] = {}
        need: Dict[int, np.ndarray] = {}
        slots = []
        for f in sorted(det_frames):
            D = det_frames[f]
            F = np.zeros((len(D), FEAT_DIM), dtype=np.float32)
            rows = self.gid_row.get(f, {})
            idx = []
            for j, g in enumerate(D.src_gid):
                g = int(g)
                if g != FP_GID and reuse and g in rows:
                    F[j] = self.gt_feat[f][rows[g]]
                    self.n_reused += 1
                else:
                    idx.append(j)
            out[f] = F
            need[f] = np.asarray(D.boxes[idx], dtype=np.float64) if idx else \
                np.zeros((0, 4), dtype=np.float64)
            slots.append((f, np.asarray(idx, dtype=int)))
        res, offsets = self._raw(need)
        feats = res["feats"]
        for f, idx in slots:
            if len(idx):
                o = offsets[f]
                out[f][idx] = feats[o:o + len(idx)]
        return out


def list_benchmark_sequences(data_root: str) -> dict:
    """Return {benchmark: [(name, dir), ...]} using the frozen dataset roles."""
    return {
        "mot17": [(os.path.basename(d.rstrip("/")), d) for d in sorted(
            glob.glob(os.path.join(data_root, "mot17/MOT17/train", "*-FRCNN")))],
        "dancetrack": [(os.path.basename(d.rstrip("/")), d) for d in sorted(
            glob.glob(os.path.join(data_root, "dancetrack/val", "dancetrack*")))],
        "mot20": [(os.path.basename(d.rstrip("/")), d) for d in sorted(
            glob.glob(os.path.join(data_root, "mot20/MOT20/train", "MOT20-*")))],
    }
