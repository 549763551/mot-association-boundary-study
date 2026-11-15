#!/usr/bin/env python3
"""Global, pre-registered configuration for the full real-benchmark experiment.

IMPORTANT: all thresholds are scanned (not tuned on the evaluation split).
The nominal operating point is chosen on a disjoint calibration subset only.
"""
import os

ROOT = os.environ.get("MOT_EXP_ROOT", "/root/mot_exp")
DATA = os.path.join(ROOT, "data")
FEAT = os.path.join(ROOT, "features")
TRACKS = os.path.join(ROOT, "tracks")
EVALDATA = os.path.join(ROOT, "eval_data")
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")

# ---------------------------------------------------------------- benchmarks
BENCHMARKS = {
    "dancetrack": dict(
        root=os.path.join(DATA, "dancetrack/val"),
        pattern="dancetrack*",
        te_benchmark="MOT17",
        te_split="val",
        detector_suffix=None,
    ),
    "mot17": dict(
        root=os.path.join(DATA, "mot17/MOT17/train"),
        pattern="*-FRCNN",
        te_benchmark="MOT17",
        te_split="train",
        detector_suffix="-FRCNN",
    ),
    "mot20": dict(
        root=os.path.join(DATA, "mot20/MOT20/train"),
        pattern="MOT20-*",
        te_benchmark="MOT20",
        te_split="train",
        detector_suffix=None,
    ),
}

# GT filter: exactly TrackEval's MOTChallenge convention (col6 zero_marked, col7 class)
GT_CONF_COL = 6
GT_CLASS_COL = 7
PEDESTRIAN_CLASS = 1

# ---------------------------------------------------------------- ReID
OSNET_FILE = os.path.join(ROOT, "exp/osnet_model.py")
WEIGHT_MSMT17 = os.path.join(ROOT, "weights/osnet_x1_0_msmt17.pth")
WEIGHT_IMAGENET = os.path.join(ROOT, "weights/osnet_x1_0_imagenet.pth")
FEAT_DIM = 512
REID_H, REID_W = 256, 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXTRACT_BATCH = int(os.environ.get("MOT_EXTRACT_BATCH", "256"))
EXTRACT_WORKERS = int(os.environ.get("MOT_EXTRACT_WORKERS", "8"))
EXTRACT_FP16 = False      # measured: ROCm autocast slower than FP32 for this model

# ---------------------------------------------------------------- tracker
IOU_GATE = 0.05           # a track/det pair is a candidate if predicted-box IoU >= this
MAX_AGE = 30              # frames a track survives without a match
TEMPLATE_PREFIX = 10      # number of confident frames used to build a clean template
TEMPLATE_EMA = 0.9        # only used in the online-update ablation
MIN_TRACK_LEN = 1

# ---------------------------------------------------------------- strategies
STRATEGIES = {
    "S0_motion":      dict(kind="motion"),
    "S1_global":      dict(kind="global"),
    "S2_countgated":  dict(kind="count", K=2),
    "S3_hist_auc":    dict(kind="proposed", confidence="hist_auc", online=False, tau=0.10),
    "S3_frame_calib": dict(kind="proposed", confidence="frame_calib", online=False, tau=0.10),
    "S3_online":      dict(kind="proposed", confidence="hist_auc", online=True, tau=0.10),
    "S3_saturated":   dict(kind="proposed", confidence="hist_auc", online=False, tau=1e9),
    "S3_no_abstain":  dict(kind="proposed", confidence="none", online=False, tau=0.10),
}

# tau grid for trigger-rate / budget curves (no tuning on eval split)
TAU_GRID = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1e9]
# appearance/motion fusion weight grid: rules out "just a bad weight" as an
# alternative explanation for any negative appearance result (pre-registered scan)
ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 0.8, 1.0]
ABSTAIN_THETA = 0.55      # fixed a priori, identical for all strategies
SPARSE_TOPK = 64          # sparse difference subspace dimensionality
SEEDS = [11, 23, 37, 53, 71]
NOISE_SIGMA = 0.01        # feature robustness perturbation

# ---------------------------------------------------------------- calibration
# disjoint sequence subsets used ONLY to pick the nominal tau (already fixed above)
CALIB_FRACTION = 0.20

# ---------------------------------------------------------------- statistics
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20261115

# Optional local / subset execution. The server leaves all of these unset, so
# the full three-benchmark configuration above is used unchanged.
_env_benchmarks = os.environ.get("MOT_BENCHMARKS")
if _env_benchmarks:
    _keep = {x.strip() for x in _env_benchmarks.split(",") if x.strip()}
    BENCHMARKS = {k: v for k, v in BENCHMARKS.items() if k in _keep}

# ---------------------------------------------------------------- pre-registered decisions
# A conclusion is "reproduced" only if these numeric conditions hold.
DECISION_RULES = {
    "inverted_u": dict(
        mid_band_delta_lower_ci_gt=0.0,
        mid_minus_low_gt=0.0,
        mid_minus_high_gt=0.0,
    ),
    "critical_notch": dict(
        hist_delta_upper_ci_lt=0.0,
        calib_delta_lower_ci_ge=0.0,
    ),
    "cost_overrun": dict(
        saturated_trigger_rate_gt=0.80,
        saturated_delta_vs_best_lt=0.0,
    ),
    "online_degradation": dict(
        online_minus_frozen_upper_ci_lt=0.0,
    ),
}
