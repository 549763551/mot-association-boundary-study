#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨平台增量镜像 /root/mot_exp/features/*.npy <-> 本机。

用法:
  python pull_features.py            # 服务器 -> 本机（实验每一步后调用）
  python pull_features.py --push     # 本机 -> 服务器（新实例恢复已有特征）
  python pull_features.py --list     # 只列差异，不传数据

为什么需要它:
  第三次实例断连时，本机只留下了 features 的「路径+尺寸清单」，35 个 .npy
  全部随实例丢失，导致每次都要重抽约 70 分钟 GPU。实测特征总量仅约 0.7 GB
  （三基准全量约 2~3 GB），回拉带宽 3.6 MB/s，一次全量同步约 3~14 分钟，
  远低于重抽成本，因此特征必须纳入「一步一备份」。

实现要点:
  * 用 Python tarfile 走 ssh 的 stdin/stdout 二进制流，不依赖 Windows 原生 tar，
    规避 Git Bash 的 /d/... 路径不兼容问题。
  * 按 (相对路径, 字节数) 比对，只传变化文件；np.save 写入中的半成品
    size 会变化，下次同步自动重传，不会被误判为完成。
  * 每个序列一个 .npy，天然支持断点续跑。
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import time

HOST = os.environ.get("MOT_SSH_HOST", "radeon-cloud")
REMOTE = "/root/mot_exp"
LOCAL = r"/path/to/mot_exp_backup/remote_root"
SUB = "features"
OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=25",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=60"]


def ssh_cmd(cmd, stdin=None, stdout=None, stderr=None):
    return subprocess.Popen(["ssh"] + OPTS + [HOST, cmd],
                            stdin=stdin, stdout=stdout, stderr=stderr)


def remote_manifest():
    cmd = "cd %s && find %s -name '*.npy' -printf '%%p\\t%%s\\n' 2>/dev/null | sort" % (REMOTE, SUB)
    p = subprocess.run(["ssh"] + OPTS + [HOST, cmd],
                       capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        raise RuntimeError("remote manifest failed (server down?): %s" % p.stderr[-300:])
    out = {}
    for line in p.stdout.splitlines():
        if "\t" not in line:
            continue
        path, size = line.rsplit("\t", 1)
        try:
            out[path.replace("\\", "/")] = int(size)
        except ValueError:
            pass
    return out


def local_manifest():
    root = os.path.join(LOCAL, SUB)
    out = {}
    if not os.path.isdir(root):
        return out
    for base, _dirs, files in os.walk(root):
        for f in files:
            if not f.endswith(".npy"):
                continue
            full = os.path.join(base, f)
            rel = os.path.relpath(full, LOCAL).replace("\\", "/")
            out[rel] = os.path.getsize(full)
    return out


def _extract_one(tf, member, path):
    try:
        tf.extract(member, path=path, filter="data")   # Python >= 3.12
    except TypeError:
        tf.extract(member, path=path)


def do_pull():
    rm = remote_manifest()
    lm = local_manifest()
    changed = sorted(p for p, s in rm.items() if lm.get(p) != s)
    if not changed:
        print("PULL_NO_CHANGE remote=%d local=%d" % (len(rm), len(lm)), flush=True)
        return 0
    mb = sum(rm[p] for p in changed) / 1e6
    print("PULL %d/%d files, %.1f MB" % (len(changed), len(rm), mb), flush=True)
    os.makedirs(LOCAL, exist_ok=True)
    p = ssh_cmd("cd %s && tar -cf - -T -" % REMOTE,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        p.stdin.write(("\n".join(changed) + "\n").encode("utf-8"))
        p.stdin.close()
        tf = tarfile.open(fileobj=p.stdout, mode="r|")
        n = 0
        for member in tf:
            if member.isfile() and member.name.startswith(SUB + "/"):
                _extract_one(tf, member, LOCAL)
                n += 1
        tf.close()
        err = p.stderr.read().decode("utf-8", "replace")[-300:]
        rc = p.wait(timeout=60)
        if rc != 0:
            print("PULL_TAR_WARN rc=%d %s" % (rc, err), flush=True)
    except Exception as e:
        print("PULL_FAIL %s: %s" % (type(e).__name__, e), flush=True)
        try:
            p.kill()
        except Exception:
            pass
        return 1
    lm2 = local_manifest()
    bad = [p_ for p_ in changed if lm2.get(p_) != rm[p_]]
    print("PULL_DONE extracted=%d verified=%d mismatched=%d" %
          (n, len(changed) - len(bad), len(bad)), flush=True)
    _write_state("pull", rm, bad)
    return 1 if bad else 0


def do_push():
    rm = remote_manifest()
    lm = local_manifest()
    changed = sorted(p for p, s in lm.items() if rm.get(p) != s)
    if not changed:
        print("PUSH_NO_CHANGE local=%d remote=%d" % (len(lm), len(rm)), flush=True)
        return 0
    mb = sum(lm[p] for p in changed) / 1e6
    print("PUSH %d files, %.1f MB" % (len(changed), mb), flush=True)
    p = ssh_cmd("cd %s && mkdir -p %s && tar -xf -" % (REMOTE, SUB),
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        tf = tarfile.open(fileobj=p.stdin, mode="w|")
        for rel in changed:
            tf.add(os.path.join(LOCAL, rel), arcname=rel, recursive=False)
        tf.close()
        try:
            p.stdin.close()
        except Exception:
            pass
        err = p.stderr.read().decode("utf-8", "replace")[-300:]
        rc = p.wait(timeout=60)
        if rc != 0:
            print("PUSH_TAR_FAIL rc=%d %s" % (rc, err), flush=True)
            return 1
    except Exception as e:
        print("PUSH_FAIL %s: %s" % (type(e).__name__, e), flush=True)
        try:
            p.kill()
        except Exception:
            pass
        return 1
    rm2 = remote_manifest()
    bad = [p_ for p_ in changed if rm2.get(p_) != lm[p_]]
    print("PUSH_DONE sent=%d mismatch=%d" % (len(changed), len(bad)), flush=True)
    return 1 if bad else 0


def _write_state(mode, manifest, bad):
    os.makedirs(LOCAL, exist_ok=True)
    state = dict(mode=mode, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                 n_remote_files=len(manifest),
                 total_bytes=sum(manifest.values()), mismatch=bad[:20])
    with open(os.path.join(LOCAL, "feature_sync_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="local -> server")
    ap.add_argument("--list", action="store_true", help="only show differences")
    args = ap.parse_args()
    try:
        rm = remote_manifest()
        lm = local_manifest()
    except Exception as e:
        print("SYNC_UNAVAILABLE %s: %s" % (type(e).__name__, e), flush=True)
        return 2
    diff = sorted(p for p, s in rm.items() if lm.get(p) != s)
    push_diff = sorted(p for p, s in lm.items() if rm.get(p) != s)
    print("remote=%d files %.2f GB | local=%d files %.2f GB" %
          (len(rm), sum(rm.values()) / 1e9, len(lm), sum(lm.values()) / 1e9), flush=True)
    print("pull_pending=%d push_pending=%d" % (len(diff), len(push_diff)), flush=True)
    if args.list:
        for p_ in diff[:10]:
            print("  pull:", p_, rm[p_])
        for p_ in push_diff[:10]:
            print("  push:", p_, lm[p_])
        return 0
    return do_push() if args.push else do_pull()


if __name__ == "__main__":
    sys.exit(main())
