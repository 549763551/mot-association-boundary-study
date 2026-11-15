#!/usr/bin/env python3
"""Server-side Phase 1 verification (run inside the venv on the GPU instance).

Checks three things with NO GPU usage and NO network traffic:
  1. the locally frozen config transfers with an intact SHA-256 seal,
  2. all injection / IDSW quality-gate self-tests pass on Python 3.12,
  3. Theta-hat recomputed from the SERVER's unpacked hf-mirror repack matches
     the Theta-hat computed on the LOCAL machine from the original MOT17.zip.
     (3) is a real cross-source data-equivalence check, not a formality.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import det_calibration as dc
import phase1_selftest as st
from run_phase1 import verify_frozen

DATA_ROOT = "/root/mot_exp/data"
TOL = 1e-6


def seq_of(label: str) -> str:
    m = re.search(r"(MOT17-\d{2}-[A-Z]+|MOT20-\d{2})", label)
    return m.group(1) if m else label


def main() -> int:
    print("=== [1/4] frozen config hash seal ===", flush=True)
    v = verify_frozen(os.path.join(HERE, "frozen_config.json"))
    print(json.dumps(v["_verify"], indent=2), flush=True)
    if not v["_verify"]["ok"]:
        print("HASH_VERIFY_FAILED")
        return 2
    print("  protocol=%s revision=%s conditions=%s" %
          (v["protocol"], v["freeze_revision"], v["n_conditions_unique"]),
          flush=True)

    print("\n=== [2/4] injection + IDSW quality gate on server Python ===", flush=True)
    s = st.run()
    json.dump(s, open(os.path.join(HERE, "phase1_selftest_server.json"), "w",
                      encoding="utf-8"), indent=2, ensure_ascii=False)
    for c in s["checks"]:
        print("  %-42s %s" % (c["name"], "PASS" if c["passed"] else "FAIL"),
              flush=True)
    print("  GATE:", s["quality_gate"], flush=True)

    print("\n=== [3/4] Theta-hat from SERVER unpacked data ===", flush=True)
    cal = dc.calibrate(dc.iter_pairs_from_root(DATA_ROOT))
    slim = dict(n_sequences=cal["n_sequences"], grid=cal["grid"],
                score_distribution=cal["score_distribution"],
                per_sequence=[{k: r[k] for k in
                               ("label", "n_gt", "n_det", "n_match",
                                "sigma_loc", "beta", "lambd", "tau")}
                              for r in cal["per_sequence"]])
    json.dump(slim, open(os.path.join(HERE, "calibration_server.json"), "w",
                         encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n=== [4/4] cross-source equivalence (local original zip vs server repack) ===",
          flush=True)
    lp = os.path.join(HERE, "calibration_report.json")
    if not os.path.exists(lp):
        print("  SKIP: local calibration_report.json not copied")
        return 0 if s["passed"] else 1
    local = json.load(open(lp, encoding="utf-8"))
    L = {seq_of(r["label"]): r for r in local["per_sequence"]}
    S = {seq_of(r["label"]): r for r in cal["per_sequence"]}
    common = sorted(set(L) & set(S))
    diffs = []
    for k in common:
        for field in ("n_gt", "n_det", "n_match", "sigma_loc", "beta",
                      "lambd", "tau"):
            a, b = float(L[k][field]), float(S[k][field])
            scale = max(1.0, abs(a), abs(b))
            if abs(a - b) / scale > TOL:
                diffs.append((k, field, a, b))
    ok = (len(common) == len(L) == len(S)) and not diffs and s["passed"]
    result = dict(
        status="CROSS_SOURCE_CALIBRATION_OK" if ok else "CROSS_SOURCE_MISMATCH",
        local_sequences=len(L), server_sequences=len(S),
        matched_sequences=len(common), n_field_diffs=len(diffs),
        diffs=diffs[:20], quality_gate=s["quality_gate"],
        hash_ok=v["_verify"]["ok"],
        grid_rule=cal["grid"]["rule"],
        frozen_digest=v["integrity"]["digest"],
        note="confirms the hf-mirror repack is equivalent to the original "
             "MOT17/MOT20 archives for Theta-hat calibration purposes",
    )
    json.dump(result, open(os.path.join(HERE, "phase1_server_report.json"), "w",
                           encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
