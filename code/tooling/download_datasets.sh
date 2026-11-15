#!/usr/bin/env bash
# 服务器端执行：从 hf-mirror 直下 15 GB 数据集
#
# 目的：15 GB 走服务器自己的公网出口，完全不经过本机，
#       因此无需从本机上传数据。
# 用法：bash /root/mot_exp/download_datasets.sh
#   SKIP_PROXY=1  强制不走代理（直连 hf-mirror）
#   curl -C - 支持断点续传，重复执行会跳过已完整的文件。
set -uo pipefail

D=/root/mot_exp/datasets
L=/root/mot_exp/logs
mkdir -p "$D" "$L"

# 平台新实例自带 egress proxy；非交互 ssh 可能丢环境变量，这里补默认地址。
if [ "${SKIP_PROXY:-0}" != "1" ] && [ -z "${http_proxy:-}" ]; then
    export http_proxy="http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128"
    export https_proxy="$http_proxy"
fi
echo "proxy=${http_proxy:-<none>}  target=$D"

dl() {
    local file="$1" tag="$2" url="$3" want="$4"
    local log="$L/dl_${tag}.log" i have
    for i in 1 2 3 4 5; do
        have=$(stat -c %s "$D/$file" 2>/dev/null || echo 0)
        if [ "$have" = "$want" ]; then
            echo "OK_SKIP $file $want" | tee -a "$log"
            return 0
        fi
        echo "ATTEMPT $i $file have=$have want=$want $(date -u)" >> "$log"
        curl -L -C - --retry 3 --retry-delay 2 --connect-timeout 30 \
             --max-time 3600 -o "$D/$file" "$url" >> "$log" 2>&1 || true
        have=$(stat -c %s "$D/$file" 2>/dev/null || echo 0)
        echo "  after attempt $i: $have" >> "$log"
    done
    have=$(stat -c %s "$D/$file" 2>/dev/null || echo 0)
    if [ "$have" = "$want" ]; then
        echo "OK $file $want" | tee -a "$log"
        return 0
    fi
    echo "FAIL $file got=$have want=$want" | tee -a "$log"
    return 1
}

echo "DOWNLOAD_START $(date -u)"
pids=()
dl "dancetrack_val.zip" "dancetrack" \
   "https://hf-mirror.com/datasets/noahcao/dancetrack/resolve/main/val.zip" 4209785614 &
pids+=($!)
dl "MOT17.zip" "mot17" \
   "https://hf-mirror.com/datasets/mrhwl123/MOT17/resolve/main/MOT17.zip" 5868011787 &
pids+=($!)
dl "MOT20.zip" "mot20" \
   "https://hf-mirror.com/datasets/mrhwl123/MOT20/resolve/main/MOT20.zip" 5028926248 &
pids+=($!)
dl "MOT20Labels.zip" "mot20labels" \
   "https://hf-mirror.com/datasets/mrhwl123/MOT20/resolve/main/MOT20Labels.zip" 19246343 &
pids+=($!)

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "DOWNLOAD_END $(date -u) rc=$rc"

python3 - <<'PY'
import os
exp = {"dancetrack_val.zip": 4209785614, "MOT17.zip": 5868011787,
       "MOT20.zip": 5028926248, "MOT20Labels.zip": 19246343}
D = "/root/mot_exp/datasets"
bad = []
for k, v in exp.items():
    got = os.path.getsize(os.path.join(D, k)) if os.path.exists(os.path.join(D, k)) else -1
    print("  %-24s %s got=%d expected=%d" % (k, "OK" if got == v else "MISMATCH", got, v))
    if got != v:
        bad.append(k)
print("  DATASETS_READY" if not bad else "  DATASET_MISMATCH " + ",".join(bad))
raise SystemExit(1 if bad else 0)
PY
exit $?
