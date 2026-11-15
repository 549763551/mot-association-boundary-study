#!/usr/bin/env bash
# 完整实验 v4：新实例重建 + 全量真基准实验 + 特征级逐步备份
#
# 用法: bash /d/mot_exp_assets/resume_v4.sh [ssh_alias]
#       SKIP_SETUP=1 bash /d/mot_exp_assets/resume_v4.sh radeon-cloud   # 同实例续跑
#       MOT_SKIP_FEATURE_PULL=1 bash /d/mot_exp_assets/resume_v4.sh radeon-cloud
#         = 精简同步模式：跳过 2~3 GB 特征镜像，只保留小结果文件。
#
# v4 的增量镜像设计：
#   * pull_features.py：跨平台 tar 流式增量镜像，按 (路径, 尺寸) 比对，
#     只传变化的 .npy；--push 可把本机已有特征推回全新实例。
#   * extract 采用「逐序列一次调用 + 每序列立即回拉」（extract_features.py
#     的 --seqs 参数，向后兼容），每个序列完成后立即同步。
#   * 续跑时先 --push 恢复本机特征，extract_one 自动 skip 已有 .npy，
#     因此跨实例续跑只补缺失序列。
#   * 数据集由服务器端公开镜像直下 15 GB，阶段幂等、逐基准拆分、
#     每阶段 checkpoint + 时间戳快照。
set -uo pipefail

HOST="${1:-radeon-cloud}"
ASSETS="/d/mot_exp_assets"
SRC="/path/to/home/WorkBuddy/2026-11-12-08-30-06/confusion-gated-reid/full_experiment"
SYN="/path/to/home/WorkBuddy/2026-11-12-08-30-06/confusion-gated-reid/results/experiment.json"
DEST="$SRC/results_from_server"
OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
      -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=60)
PY="cd /root/mot_exp/exp && source /root/mot_exp/venv/bin/activate && python -u"
PY3="cd /root/mot_exp/exp && python3 -u"
BENCHMARKS=(dancetrack mot17 mot20)
SKIP_SETUP="${SKIP_SETUP:-0}"
export MOT_SSH_HOST="$HOST"

mkdir -p "$DEST" "$DEST/logs" "$ASSETS/stage_logs"

sync_features() {
  if [ "${MOT_SKIP_FEATURE_PULL:-0}" = "1" ]; then
    echo "--- [feature-sync] SKIP (MOT_SKIP_FEATURE_PULL=1, minimal sync) ---"
    return 0
  fi
  echo "--- [feature-sync] 服务器 -> 本机 ---"
  python "$ASSETS/pull_features.py" || echo "  warn: feature pull failed (mirror intact)"
}

pull() {
  local label="$1"
  echo "=== [pull:$label] scp 回拉 results/ 与 logs/ ==="
  scp -r "${OPTS[@]}" "$HOST":/root/mot_exp/results/. "$DEST/" 2>/dev/null \
    || echo "  warn: results pull failed"
  scp -r "${OPTS[@]}" "$HOST":/root/mot_exp/logs/. "$DEST/logs/" 2>/dev/null \
    || echo "  warn: logs pull failed"
  ssh "${OPTS[@]}" "$HOST" \
    'find /root/mot_exp/features -name "*.npy" -printf "%p %s\n" 2>/dev/null | sort' \
    > "$DEST/feature_inventory.txt" 2>/dev/null || true
  ls -1 "$DEST" | head -15
}

checkpoint() {
  local label="$1"
  echo "=== [checkpoint:$label] 一步一备份 ==="
  pull "$label"
  python "$ASSETS/backup_experiment.py" || echo "  warn: incremental backup pass failed"
  echo "=== [checkpoint:$label] DONE ==="
}

fail() {
  echo "!!! FATAL at stage: $1"
  sync_features || true
  checkpoint "failed_$1" || true
  exit 1
}

run_stage() {
  local label="$1"; shift
  local cmd="$1"
  local log="$ASSETS/stage_logs/$label.log"
  echo "=== [stage:$label] ==="
  ssh "${OPTS[@]}" "$HOST" "$cmd" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" != "0" ]; then
    echo "  stage $label exit=$rc (see $log)"
    return "$rc"
  fi
  return 0
}

echo "############################################################"
echo "# v4 START $(date)"
echo "############################################################"
echo "=== 本机已有特征 ==="
python "$ASSETS/pull_features.py" --list 2>/dev/null | head -4 || true

if [ "$SKIP_SETUP" != "1" ]; then
  echo "=== [0] 前置检查：数据集尺寸 / 解压工具 / ROCm wheels ==="
  ssh "${OPTS[@]}" "$HOST" 'bash -s' <<'PRE' || fail "preflight"
set -e
python3 - <<'PY'
import os
exp = {"dancetrack_val.zip": 4209785614, "MOT17.zip": 5868011787,
       "MOT20.zip": 5028926248, "MOT20Labels.zip": 19246343}
D = "/root/mot_exp/datasets"
bad = []
for k, v in exp.items():
    got = os.path.getsize(os.path.join(D, k)) if os.path.exists(os.path.join(D, k)) else -1
    print("  %-24s %s got=%d expected=%d" % (k, "OK" if got == v else "MISMATCH", got, v))
    if got != v: bad.append(k)
print("  DATASETS_OK" if not bad else "  DATASET_MISMATCH " + ",".join(bad))
raise SystemExit(1 if bad else 0)
PY
command -v unzip >/dev/null || { echo "  installing unzip"; apt-get install -y -qq unzip >/dev/null 2>&1; }
echo "  unzip: $(command -v unzip)"
ls -1 /torch-2.9.1+rocm7.2.1*.whl /torchvision-0.24.0+rocm7.2.1*.whl /triton-3.5.1+rocm7.2.1*.whl
test -f /root/mot_exp/code/TrackEval/trackeval/__init__.py
test -f /root/mot_exp/code/deep-person-reid/torchreid/models/osnet.py
test -f /root/mot_exp/weights/osnet_x1_0_msmt17.pth
echo "PREFLIGHT_OK"
PRE

  echo "=== [1] 推送实验代码与部署脚本 ==="
  ssh "${OPTS[@]}" "$HOST" 'mkdir -p /root/mot_exp/exp /root/mot_exp/results /root/mot_exp/logs /root/mot_exp/features' || fail "mkdir"
  scp "${OPTS[@]}" "$SRC"/*.py "$HOST":/root/mot_exp/exp/ || fail "push_code"
  scp "${OPTS[@]}" "$SYN" "$HOST":/root/mot_exp/exp/synthetic_experiment.json || fail "push_synth"
  scp "${OPTS[@]}" "$ASSETS/server_setup.sh" "$ASSETS/MANIFEST.md" "$HOST":/root/mot_exp/ || fail "push_setup"
  ssh "${OPTS[@]}" "$HOST" \
    'cp /root/mot_exp/code/deep-person-reid/torchreid/models/osnet.py /root/mot_exp/exp/osnet_model.py' \
    || fail "push_osnet"
  run_stage "py_compile" "$PY3 -c 'import exp_config as c; print(\"ROOT\", c.ROOT); print(\"BENCHMARKS\", list(c.BENCHMARKS)); import py_compile, glob; [py_compile.compile(f, doraise=True) for f in glob.glob(\"/root/mot_exp/exp/*.py\")]; print(\"PY_COMPILE_OK\")'" \
    || fail "py_compile"
  checkpoint "code_synced"

  echo "=== [2] venv + 依赖 + ROCm PyTorch ==="
  run_stage "venv" 'set -e; cd /root/mot_exp
[ -x venv/bin/python ] || python3 -m venv venv
source venv/bin/activate
DEPS="numpy scipy opencv-python-headless matplotlib pandas tqdm pillow pyyaml tabulate filelock typing-extensions sympy networkx jinja2 fsspec mpmath markupsafe"
pip install -q --disable-pip-version-check --timeout 60 --retries 5 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple $DEPS \
  || pip install -q --no-index --find-links /root/mot_exp/wheels $DEPS
pip install -q --no-deps /torch-2.9.1+rocm7.2.1*.whl /torchvision-0.24.0+rocm7.2.1*.whl /triton-3.5.1+rocm7.2.1*.whl
python -c "import torch, numpy, cv2, scipy, pandas, matplotlib; print(\"torch\", torch.__version__); print(\"hip\", getattr(torch.version, \"hip\", None)); print(\"gpu_ready\", torch.cuda.is_available()); assert torch.cuda.is_available()"
echo VENV_OK' || fail "venv"
  checkpoint "venv_ready"

  echo "=== [3] 解压 + 目录归一化 + GT 统计 ==="
  run_stage "setup" 'bash /root/mot_exp/server_setup.sh' || fail "setup"
  checkpoint "data_unpacked"

  echo "=== [4] 序列发现校验 ==="
  run_stage "seq_gate" "$PY -c '
from data_utils import list_sequences
exp = {\"dancetrack\": 25, \"mot17\": 7, \"mot20\": 4}
bad = []
for b, n in exp.items():
    s = list_sequences(b)
    print(\"  %-12s %d sequences %s\" % (b, len(s), [x[0] for x in s[:3]]))
    if len(s) != n: bad.append(b)
print(\"SEQ_GATE_OK\" if not bad else \"SEQ_GATE_FAIL \" + \",\".join(bad))
raise SystemExit(1 if bad else 0)
'" || fail "seq_gate"
  checkpoint "seq_gate"
else
  echo "=== SKIP_SETUP=1: 复用同实例环境，跳过 [0]-[4] ==="
fi

echo "=== [5a] 把本机已有特征推回服务器（跨实例续跑的关键） ==="
if [ "${MOT_SKIP_FEATURE_PULL:-0}" = "1" ]; then
  echo "  SKIP (MOT_SKIP_FEATURE_PULL=1: feature mirroring disabled)"
else
  python "$ASSETS/pull_features.py" --push || echo "  warn: feature push failed, will re-extract from scratch"
fi
checkpoint "features_restored"

echo "=== [5b] ReID 特征抽取：逐序列调用 + 每序列立即回拉本机 ==="
for b in "${BENCHMARKS[@]}"; do
  seqs=$(ssh "${OPTS[@]}" "$HOST" \
    "$PY -c 'from data_utils import list_sequences; import sys; print(\" \".join(n for n,_ in list_sequences(sys.argv[1])))' $b" \
    2>/dev/null | tail -1)
  if [ -z "$seqs" ]; then fail "list_seqs_$b"; fi
  echo "--- $b: $seqs"
  for s in $seqs; do
    run_stage "extract_${b}_${s}" \
      "$PY extract_features.py --benchmarks $b --seqs $s --fp16 0 --workers 8" \
      || fail "extract_${b}_${s}"
    sync_features
  done
  checkpoint "extract_$b"
done

echo "=== [6] 特征可分性诊断 ==="
run_stage "diagnose" "$PY diagnose_features.py" || fail "diagnose"
checkpoint "diagnose"

echo "=== [7] 全量跟踪（逐基准，run_stats.json 增量合并） ==="
for b in "${BENCHMARKS[@]}"; do
  run_stage "track_$b" "$PY run_full_experiment.py --stage track --benchmarks $b" \
    || fail "track_$b"
  checkpoint "track_$b"
done

echo "=== [8] 官方 TrackEval 评估（逐基准，trackeval_results.json 增量合并） ==="
for b in "${BENCHMARKS[@]}"; do
  run_stage "eval_$b" "$PY run_full_experiment.py --stage eval --benchmarks $b" \
    || fail "eval_$b"
  checkpoint "eval_$b"
done

echo "=== [9] 分带 / bootstrap / 四条预注册判定 ==="
run_stage "analyze" "$PY run_full_experiment.py --stage analyze" || fail "analyze"
checkpoint "analyze"

echo "=== [10] 合成 vs 真数据差异报告 ==="
run_stage "diff_report" "$PY make_diff_report.py" || fail "diff_report"
checkpoint "diff_report"

echo "=== [11] 最终判定摘要 ==="
python - <<'PY'
import json, os
p = r"/path/to/home/WorkBuddy/2026-11-12-08-30-06/confusion-gated-reid/full_experiment/results_from_server/full_experiment_report.json"
if not os.path.exists(p):
    print("NO_FINAL_REPORT (全量判定未完成)")
else:
    r = json.load(open(p, encoding="utf-8"))
    print("decisions:", r["decisions"])
    print("all_conclusions_reproduced:", r["all_conclusions_reproduced"])
    print("idsw_consistency:", r.get("idsw_consistency"))
    print("BRANCH:", "all conclusions reproduced" if r["all_conclusions_reproduced"]
          else "difference report recorded")
PY

echo "############################################################"
echo "# v4 DONE $(date)  results=$DEST"
echo "############################################################"
