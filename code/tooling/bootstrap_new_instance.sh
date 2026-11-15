#!/usr/bin/env bash
# 新实例资产引导（省本机流量版）
#
# 解决的问题：resume_v4.sh 的 [0] preflight 假设 datasets/code/weights 已在服务器上，
# 而「服务器侧直下 15 GB」的脚本此前只存在于被销毁的实例上。
# 全新空白实例必须先跑本脚本，再跑 resume_v4.sh。
#
# 用法: bash /d/mot_exp_assets/bootstrap_new_instance.sh [ssh_alias]
# 前提: ~/.ssh/config 中该别名的 Port 已改为新实例端口。
#
# 流量开关（默认即最省流量）：
#   PUSH_WEIGHTS=0   权重 27 MB 也改由服务器下载（默认 1，本机推 27 MB）
#   PUSH_WHEELS=1    额外推 152 MB 离线 wheels（默认 0；v4 首选在线清华源）
#   SKIP_DOWNLOAD=1  服务器上已有 15 GB zip 时跳过下载
#   SKIP_PROXY=1     下载时强制直连 hf-mirror
#
# 本机流量账：默认约 27.5 MB 上行；15 GB 数据集由服务器自下，0 本机流量。
set -uo pipefail

HOST="${1:-radeon-cloud}"
ASSETS="/d/mot_exp_assets"
OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
      -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=60)
PUSH_WEIGHTS="${PUSH_WEIGHTS:-1}"
PUSH_WHEELS="${PUSH_WHEELS:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_PROXY="${SKIP_PROXY:-0}"

echo "############################################################"
echo "# BOOTSTRAP START $(date)  host=$HOST"
echo "############################################################"

echo "=== [0] 连通性 / 平台预置检查 ==="
ssh "${OPTS[@]}" "$HOST" 'bash -s' <<'PRE' || { echo "SSH_UNREACHABLE：请确认新实例端口已写入 ~/.ssh/config"; exit 1; }
set -e
mkdir -p /root/mot_exp/datasets /root/mot_exp/code /root/mot_exp/weights \
         /root/mot_exp/exp /root/mot_exp/results /root/mot_exp/logs \
         /root/mot_exp/features /root/mot_exp/wheels
python3 -V
echo "  proxy=${http_proxy:-<none>}"
ls -1 /torch-2.9.1+rocm7.2.1*.whl /torchvision-0.24.0+rocm7.2.1*.whl \
      /triton-3.5.1+rocm7.2.1*.whl 2>/dev/null \
    || echo "  WARN_ROCM_WHEELS_MISSING（平台未预置，需另找 ROCm torch 安装源）"
command -v unzip >/dev/null || { apt-get install -y -qq unzip >/dev/null 2>&1 || true; }
echo "  unzip: $(command -v unzip || echo MISSING)"
echo "PRECHECK_OK"
PRE

echo "=== [1] 推小文件（源码包 0.5 MB + 脚本 + 权重 27 MB）==="
scp "${OPTS[@]}" "$ASSETS/download_datasets.sh" "$ASSETS/server_setup.sh" \
    "$ASSETS/MANIFEST.md" "$HOST":/root/mot_exp/ || { echo "PUSH_SCRIPTS_FAILED"; exit 1; }
scp "${OPTS[@]}" "$ASSETS/code/TrackEval.tar.gz" "$ASSETS/code/deep-person-reid.tar.gz" \
    "$HOST":/root/mot_exp/code/ || { echo "PUSH_CODE_FAILED"; exit 1; }
if [ "$PUSH_WEIGHTS" = "1" ]; then
    scp "${OPTS[@]}" "$ASSETS"/weights/*.pth "$HOST":/root/mot_exp/weights/ \
        || { echo "PUSH_WEIGHTS_FAILED"; exit 1; }
else
    ssh "${OPTS[@]}" "$HOST" 'cd /root/mot_exp/weights && \
      curl -L -o osnet_x1_0_msmt17.pth \
        "https://hf-mirror.com/kaiyangzhou/osnet/resolve/main/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth" && \
      curl -L -o osnet_x1_0_imagenet.pth \
        "https://hf-mirror.com/kaiyangzhou/osnet/resolve/main/osnet_x1_0_imagenet.pth"' \
        || { echo "SERVER_WEIGHT_DOWNLOAD_FAILED"; exit 1; }
fi
if [ "$PUSH_WHEELS" = "1" ]; then
    scp -r "${OPTS[@]}" "$ASSETS"/wheels/. "$HOST":/root/mot_exp/wheels/ \
        || echo "  warn: wheels push failed (在线 pip 仍可用)"
fi

echo "=== [1b] 展开源码并校验 ==="
ssh "${OPTS[@]}" "$HOST" 'bash -s' <<'EXPAND' || { echo "CODE_EXPAND_FAILED"; exit 1; }
set -e
cd /root/mot_exp/code
if [ ! -f TrackEval/trackeval/__init__.py ]; then
    mkdir -p TrackEval
    tar xzf TrackEval.tar.gz -C TrackEval --strip-components=1
fi
if [ ! -f deep-person-reid/torchreid/models/osnet.py ]; then
    mkdir -p deep-person-reid
    tar xzf deep-person-reid.tar.gz -C deep-person-reid --strip-components=1
fi
chmod +x /root/mot_exp/download_datasets.sh /root/mot_exp/server_setup.sh
test -f /root/mot_exp/code/TrackEval/trackeval/__init__.py
test -f /root/mot_exp/code/deep-person-reid/torchreid/models/osnet.py
echo "CODE_EXPANDED_OK"
EXPAND

echo "=== [2] 服务器侧直下 15 GB（不经过本机，0 流量）==="
if [ "$SKIP_DOWNLOAD" = "1" ]; then
    echo "  SKIP_DOWNLOAD=1，跳过"
else
    ssh "${OPTS[@]}" "$HOST" "cd /root/mot_exp && SKIP_PROXY=$SKIP_PROXY bash download_datasets.sh" \
        || { echo "DOWNLOAD_FAILED：看 /root/mot_exp/logs/dl_*.log，可改 SKIP_PROXY=1 重试"; exit 1; }
fi

echo "=== [3] 资产终检（与 v4 preflight 同口径）==="
ssh "${OPTS[@]}" "$HOST" 'bash -s' <<'CHK' || { echo "ASSET_CHECK_FAILED"; exit 1; }
set -e
python3 - <<'PY'
import os
exp = {"dancetrack_val.zip": (4209785614,), "MOT17.zip": (5860214001, 5868011787),
       "MOT20.zip": (5028926248,), "MOT20Labels.zip": (19246343,)}
w = {"osnet_x1_0_msmt17.pth": 17273805, "osnet_x1_0_imagenet.pth": 10910553}
bad = []
for k, sizes in exp.items():
    p = "/root/mot_exp/datasets/" + k
    got = os.path.getsize(p) if os.path.exists(p) else -1
    print("  %-24s %s got=%d" % (k, "OK" if got in sizes else "MISMATCH", got))
    if got not in sizes: bad.append(k)
for k, size in w.items():
    p = "/root/mot_exp/weights/" + k
    got = os.path.getsize(p) if os.path.exists(p) else -1
    print("  %-24s %s got=%d" % (k, "OK" if got == size else "MISMATCH", got))
    if got != size: bad.append(k)
print("  BOOTSTRAP_OK" if not bad else "  BOOTSTRAP_INCOMPLETE " + ",".join(bad))
raise SystemExit(1 if bad else 0)
PY
CHK

echo "############################################################"
echo "# BOOTSTRAP DONE $(date)"
echo "# 下一步（流量模式，关掉 2~3 GB 特征回拉）："
echo "#   MOT_SKIP_FEATURE_PULL=1 bash /d/mot_exp_assets/resume_v4.sh $HOST"
echo "# 若之后换成 WiFi / 不限流量，去掉该变量即可恢复逐步特征备份。"
echo "############################################################"
