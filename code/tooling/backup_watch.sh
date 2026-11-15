#!/usr/bin/env bash
# 持续增量备份守护：每 20 秒做一次「一步一备份」，直到最终报告落盘。
# 与服务器上正在跑的 run_phase2.py 完全独立，不改动、不干扰它。
# 用法: bash /d/mot_exp_assets/backup_watch.sh
# 2026-11-15: END_MARK 从原实验的 difference_report.md 改为噪声实验的
# final_report.md；锁已改为 OS 租约（不再删除 .backup.lock），避免死锁。
set -uo pipefail
SCRIPT="/path/to/mot_exp_assets/backup_experiment.py"
BASE="/path/to/mot_exp_backup"
END_MARK="$BASE/mirror/exp_noise/results/final_report.md"
INTERVAL="${1:-20}"

mkdir -p "$BASE"
echo "[$(date '+%F %T')] BACKUP_DAEMON_START interval=${INTERVAL}s end_mark=$END_MARK" \
  >> "$BASE/daemon.log"

for i in $(seq 1 3000); do
  python "$SCRIPT" >> "$BASE/daemon.log" 2>&1 || true
  if [ -f "$END_MARK" ]; then
    sleep "$INTERVAL"
    python "$SCRIPT" >> "$BASE/daemon.log" 2>&1 || true
    echo "[$(date '+%F %T')] BACKUP_DAEMON_DONE final report backed up" \
      >> "$BASE/daemon.log"
    exit 0
  fi
  sleep "$INTERVAL"
done
echo "[$(date '+%F %T')] BACKUP_DAEMON_TIMEOUT" >> "$BASE/daemon.log"
