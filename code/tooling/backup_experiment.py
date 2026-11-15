#!/usr/bin/env python3
"""One incremental backup pass of the remote MOT experiment.

Designed for an unstable remote instance: every pass
  1. inventories every produced file on the server (path + size),
  2. compares against the local mirror,
  3. streams ONLY new / changed files back via a single tar pipe,
  4. writes a timestamped snapshot whenever anything changed.

Everything is append-safe: a vanished server never deletes the local mirror.
"""
import os
import sys
import json
import time
import shutil
import subprocess
from datetime import datetime

HOST = os.environ.get("MOT_SSH_HOST", "radeon-cloud")
BASE = "/path/to/mot_exp_backup"
MIRROR = BASE + "/mirror"
SNAP = BASE + "/snapshots"
LOG = BASE + "/backup_log.txt"
STATE = BASE + "/backup_state.json"
LIST = BASE + "/.need_list"
REMOTE_ROOT = "/root/mot_exp"
DIRS = ("results", "logs", "exp", "exp_noise", "tracks", "eval_data", "features")
OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20"]

# This pass mirrors conclusion-bearing small files (results/logs/exp/eval CSV)
# and keeps a path+size inventory of everything else. Per-tracker MOT result
# txt (~2 GB) are recomputable and stay inventory-only. Feature .npy are NO
# LONGER handled here: v4 mirrors them separately with pull_features.py,
# which streams each completed sequence back immediately (one .npy per
# sequence, size-stable, ~0.7-3 GB total). The third instance outage showed
# that inventory-only features cost ~70 min of GPU re-extraction every time.
SKIP_SUFFIXES = (".npy", ".pth", ".zip", ".whl", ".tar.gz", ".bin")
MAX_BYTES = 2 * 1024 * 1024
SKIP_INVENTORY = BASE + "/large_file_inventory.txt"


def bpath(p):
    """D:/x -> /d/x for Git Bash scp/tar."""
    if len(p) > 2 and p[1] == ":" :
        return "/" + p[0].lower() + p[2:]
    return p


def log(msg):
    line = "[%s] %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ssh(cmd, timeout=90):
    return subprocess.run(["ssh"] + OPTS + [HOST, cmd],
                          capture_output=True, text=True, timeout=timeout)


def remote_inventory():
    cmd = ("cd %s 2>/dev/null && find %s -type f -printf '%%p\\t%%s\\n' 2>/dev/null"
           % (REMOTE_ROOT, " ".join(DIRS)))
    r = ssh(cmd)
    if r.returncode != 0 and not r.stdout:
        return None
    out = {}
    skipped = {}
    for line in r.stdout.splitlines():
        if "\t" not in line:
            continue
        p, s = line.rsplit("\t", 1)
        try:
            size = int(s)
        except ValueError:
            continue
        p = p.replace("\\", "/")
        if (size > MAX_BYTES or p.endswith(SKIP_SUFFIXES)
                or "/__pycache__/" in p or p.endswith(".pyc")):
            skipped[p] = size
            continue
        out[p] = size
    try:
        with open(SKIP_INVENTORY, "w", encoding="utf-8") as f:
            f.write("# excluded from mirror: recomputable from /path/to/mot_exp_assets raw data\n")
            for p in sorted(skipped):
                f.write("%s\t%d\n" % (p, skipped[p]))
    except OSError:
        pass
    return out


def local_inventory():
    out = {}
    for d in DIRS:
        root = os.path.join(MIRROR, d)
        if not os.path.isdir(root):
            continue
        for base, _, files in os.walk(root):
            for fn in files:
                full = os.path.join(base, fn)
                rel = os.path.relpath(full, MIRROR).replace("\\", "/")
                try:
                    out[rel] = os.path.getsize(full)
                except OSError:
                    pass
    return out


def stage_of(paths):
    tags = set()
    for p in paths:
        if p.startswith("features/"):
            tags.add("extract_features")
        elif p.startswith("tracks/") or ("/data/" in p and p.endswith(".txt")):
            tags.add("track")
        elif p.endswith("run_stats.json"):
            tags.add("track")
        elif p.endswith("trackeval_results.json") or "pedestrian_detailed.csv" in p:
            tags.add("eval")
        elif p.endswith("full_experiment_report.json") or p.endswith("full_experiment_figures.png"):
            tags.add("analyze")
        elif p.endswith("difference_report.md"):
            tags.add("diff_report")
        elif p.endswith("feature_diagnostics.json"):
            tags.add("diagnose")
        elif p.startswith("logs/"):
            tags.add("log")
        elif p.startswith("exp_noise/"):
            tags.add("phase2")
        elif p.startswith("exp/"):
            tags.add("code")
        else:
            tags.add("other")
    return ",".join(sorted(tags))


def stream_changed(need):
    """Push the relative-path list to the server, then untar changed files locally."""
    with open(LIST, "w", encoding="utf-8", newline="\n") as f:
        for p in need:
            f.write(p + "\n")
    r = subprocess.run(["scp"] + OPTS + [bpath(LIST), HOST + ":/tmp/mot_backup_list.txt"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        log("  ERROR pushing file list: %s" % r.stderr.strip()[:200])
        return False
    tar_cmd = "cd %s && tar -cf - -T /tmp/mot_backup_list.txt 2>/dev/null" % REMOTE_ROOT
    p1 = subprocess.Popen(["ssh"] + OPTS + [HOST, tar_cmd],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(["tar", "-xf", "-", "-C", MIRROR],
                          stdin=p1.stdout, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=True)
    p1.stdout.close()
    _, err = p2.communicate(timeout=900)
    rc1 = p1.wait()
    if p2.returncode != 0 or rc1 != 0:
        log("  ERROR tar stream rc_ssh=%s rc_tar=%s %s" % (rc1, p2.returncode, (err or "")[:200]))
        return False
    return True


def snapshot(need, n_remote):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(SNAP, ts)
    os.makedirs(dst, exist_ok=True)
    for d in ("results", "logs", "exp", "exp_noise"):
        src = os.path.join(MIRROR, d)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dst, d), dirs_exist_ok=True)
    with open(os.path.join(dst, "_change.json"), "w", encoding="utf-8") as f:
        json.dump(dict(timestamp=ts, stage=stage_of(need), n_changed=len(need),
                       changed=need[:200], n_remote_files=n_remote), f,
                  ensure_ascii=False, indent=2)
    return dst


LOCK = BASE + "/.backup.lock"

# The lock file is a persistent *lease* file and is NEVER deleted.
# Reason: the sandbox safe-delete guard blocked os.remove() once the per-turn
# deletion counter hit its threshold, so release_lock() silently failed, the
# O_EXCL lock could never be re-acquired and the daemon deadlocked at
# 2026-11-15 12:34 UTC while the experiment kept producing cells.
# An OS-held advisory lock is released automatically even if a pass is killed,
# so no stale-lock timeout or delete is needed.
try:
    import msvcrt

    def _try_flock(fd):
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _funlock(fd):
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
except ImportError:  # POSIX fallback
    import fcntl

    def _try_flock(fd):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _funlock(fd):
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


def acquire_lock():
    """Serialise daemon ticks and explicit checkpoints; return fd or None.

    Byte 0 of the lease file is the locked byte and is never rewritten;
    the human-readable owner record lives from byte 1 onwards.
    """
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o644)
        if os.fstat(fd).st_size < 1:
            os.write(fd, b"L")
        os.lseek(fd, 0, os.SEEK_SET)
        if not _try_flock(fd):
            os.close(fd)
            return None
        os.ftruncate(fd, 1)
        os.lseek(fd, 1, os.SEEK_SET)
        os.write(fd, ("RUNNING pid=%d at=%s" %
                      (os.getpid(), datetime.now().isoformat(timespec="seconds"))
                      ).encode())
        return fd
    except OSError:
        try:
            os.close(fd)
        except (OSError, UnboundLocalError):
            pass
        return None


def release_lock(fd):
    try:
        os.ftruncate(fd, 1)
        os.lseek(fd, 1, os.SEEK_SET)
        os.write(fd, ("FREE pid=%d at=%s" %
                      (os.getpid(), datetime.now().isoformat(timespec="seconds"))
                      ).encode())
    except OSError:
        pass
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        _funlock(fd)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def main():
    fd = acquire_lock()
    if fd is None:
        log("BUSY (another backup pass is running), skipping this tick")
        return 0
    try:
        return _run()
    finally:
        release_lock(fd)


def _run():
    for d in DIRS:
        os.makedirs(os.path.join(MIRROR, d), exist_ok=True)
    os.makedirs(SNAP, exist_ok=True)

    remote = remote_inventory()
    if remote is None:
        log("SERVER_UNREACHABLE (mirror intact, %d files kept)" % len(local_inventory()))
        return 2

    local = local_inventory()
    need = sorted(p for p, s in remote.items() if local.get(p) != s)
    state = dict(last_check=datetime.now().isoformat(timespec="seconds"),
                 n_remote_files=len(remote), n_local_files=len(local),
                 n_changed=len(need), last_stage=stage_of(need) if need else None)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    if not need:
        log("no change (%d remote files mirrored)" % len(remote))
        return 0

    stages = stage_of(need)
    log("STEP BACKUP stage=%s changed=%d (e.g. %s)" %
        (stages, len(need), ", ".join(need[:3])))
    ok = stream_changed(need)
    if not ok:
        return 1
    # verify
    local2 = local_inventory()
    missing = [p for p in need if local2.get(p) != remote[p]]
    if missing:
        log("  WARNING %d files still missing locally: %s" % (len(missing), missing[:3]))
    else:
        log("  verified: all %d changed files present locally" % len(need))
    dst = snapshot(need, len(remote))
    log("  snapshot -> %s" % dst)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log("BACKUP_EXCEPTION %s: %s" % (type(e).__name__, e))
        sys.exit(1)
