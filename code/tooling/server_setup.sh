#!/usr/bin/env bash
# 完整实验 · 服务器端资产校验与解压
# 用法: bash /root/mot_exp/server_setup.sh
set -euo pipefail

ROOT=/root/mot_exp
DATA="$ROOT/data"
source "$ROOT/venv/bin/activate"

echo "=== [1/5] Python / ROCm 环境自检 ==="
python - <<'PY'
import sys
import numpy, cv2, scipy, pandas, matplotlib, torch, torchvision
print("numpy      ", numpy.__version__)
print("cv2        ", cv2.__version__)
print("scipy      ", scipy.__version__)
print("pandas     ", pandas.__version__)
print("torch      ", torch.__version__)
print("torchvision", torchvision.__version__)
print("hip        ", getattr(torch.version, "hip", None))
print("gpu_ready  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_name   ", torch.cuda.get_device_name(0))
sys.path.insert(0, "/root/mot_exp/code/TrackEval")
sys.path.insert(0, "/root/mot_exp/code/deep-person-reid")
try:
    import trackeval
    print("TrackEval  ", trackeval.__file__)
except Exception as e:
    print("TrackEval IMPORT_FAIL", type(e).__name__, e)
PY

echo "=== [2/5] 资产大小校验 ==="
python - <<'PY'
import os
# MOT17.zip accepts two byte-identical-content sources:
#   5860214001 = motchallenge.net original (used on the first instance)
#   5868011787 = hf-mirror mrhwl123/MOT17 repack; same 33705 jpg / 21 gt.txt,
#                plus labels_with_ids/ YOLO labels and an extra images/ level.
# MOT17Labels.zip is optional: the hf-mirror repack already contains gt.txt.
exp = {
    "dancetrack_val.zip": (4209785614,),
    "MOT17.zip": (5860214001, 5868011787),
    "MOT20.zip": (5028926248,),
    "MOT20Labels.zip": (19246343,),
}
optional = {"MOT17Labels.zip": (10107022,)}
root = "/root/mot_exp/datasets"
ok = True
for name, sizes in exp.items():
    p = os.path.join(root, name)
    got = os.path.getsize(p) if os.path.exists(p) else -1
    flag = "OK" if got in sizes else "MISMATCH"
    if flag != "OK":
        ok = False
    print(f"  {flag:9s} {name:24s} got={got} expected={sizes}")
for name, sizes in optional.items():
    p = os.path.join(root, name)
    if not os.path.exists(p):
        print(f"  SKIP      {name:24s} (optional, GT already in MOT17.zip)")
        continue
    got = os.path.getsize(p)
    flag = "OK" if got in sizes else "MISMATCH"
    if flag != "OK":
        ok = False
    print(f"  {flag:9s} {name:24s} got={got} expected={sizes}")
wroot = "/root/mot_exp/weights"
for name, size in {"osnet_x1_0_msmt17.pth": 17273805,
                   "osnet_x1_0_imagenet.pth": 10910553}.items():
    p = os.path.join(wroot, name)
    got = os.path.getsize(p) if os.path.exists(p) else -1
    flag = "OK" if got == size else "MISMATCH"
    if got != size:
        ok = False
    print(f"  {flag:9s} {name:24s} got={got} expected={size}")
print("  ALL_ASSETS_OK" if ok else "  ASSET_CHECK_FAILED")
PY

echo "=== [3/5] 解压数据集 ==="
mkdir -p "$DATA/dancetrack" "$DATA/mot17" "$DATA/mot20"
unzip_one() {
    local zip="$1" dest="$2" tag="$3"
    if [ ! -f "$zip" ]; then echo "  SKIP missing $zip"; return; fi
    if [ -f "$dest/.unpacked" ]; then echo "  SKIP already unpacked: $tag"; return; fi
    echo "  unpacking $tag ..."
    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$zip" -d "$dest" || { echo "  UNZIP_FAIL $tag"; return 1; }
    else
        python -m zipfile -e "$zip" "$dest" || { echo "  UNZIP_FAIL $tag"; return 1; }
    fi
    touch "$dest/.unpacked"
    echo "  done $tag"
}
unzip_one "$ROOT/datasets/dancetrack_val.zip" "$DATA/dancetrack" "DanceTrack-val"
unzip_one "$ROOT/datasets/MOT17.zip"          "$DATA/mot17"      "MOT17"
unzip_one "$ROOT/datasets/MOT20.zip"          "$DATA/mot20"      "MOT20"

echo "=== [3b/5] 目录归一化 ==="
# hf-mirror 的 mrhwl123/MOT17 重打包版把序列放在 MOT17/images/train/ 下，
# 而 exp_config.py 期望 MOT17/train/*-FRCNN。用符号链接对齐，不复制数据。
if [ -d "$DATA/mot17/MOT17/images/train" ] && [ ! -e "$DATA/mot17/MOT17/train" ]; then
    ln -s "$DATA/mot17/MOT17/images/train" "$DATA/mot17/MOT17/train"
    echo "  normalized MOT17/images/train -> MOT17/train"
else
    echo "  MOT17 layout already standard (or not yet unpacked)"
fi
# 若 GT 缺失，解标注包兜底
if [ "$(find "$DATA/mot17" -name gt.txt | wc -l)" -eq 0 ] && [ -f "$ROOT/datasets/MOT17Labels.zip" ]; then
    echo "  unpacking MOT17Labels.zip (GT fallback) ..."
    unzip -q -o "$ROOT/datasets/MOT17Labels.zip" -d "$DATA/mot17" || true
fi
if [ "$(find "$DATA/mot20" -name gt.txt | wc -l)" -eq 0 ] && [ -f "$ROOT/datasets/MOT20Labels.zip" ]; then
    echo "  unpacking MOT20Labels.zip (GT fallback) ..."
    unzip -q -o "$ROOT/datasets/MOT20Labels.zip" -d "$DATA/mot20" || true
fi

echo "=== [4/5] 目录结构与 GT 统计 ==="
for d in dancetrack mot17 mot20; do
    echo "--- $d (depth<=3) ---"
    find "$DATA/$d" -maxdepth 3 -type d | head -20
    echo "  gt.txt count : $(find "$DATA/$d" -name 'gt.txt' | wc -l)"
    echo "  seqinfo count: $(find "$DATA/$d" -name 'seqinfo.ini' | wc -l)"
    echo "  image count  : $(find "$DATA/$d" -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)"
done

echo "=== [5/5] ReID 权重可加载性 ==="
python - <<'PY'
import torch, os
for name in ["osnet_x1_0_msmt17.pth", "osnet_x1_0_imagenet.pth"]:
    p = os.path.join("/root/mot_exp/weights", name)
    try:
        sd = torch.load(p, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        print(f"  OK  {name}: {len(sd)} tensors, sample key={next(iter(sd))[:60]}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
PY

echo "=== SETUP_DONE ==="
