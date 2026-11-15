# 05 · Reproduction, Integrity and Archive Status

[Back to README](../../README.md) · [中文](../zh/05-reproduction-and-archive.md)

## 1. Archive status

- Experiment A was terminated before full evaluation; Experiment B's confirm stage was not launched. `final_decision` is `null` for both.
- This repository is a static archive. Its material is a "stage conclusion plus directional tendency"; formal H1/H2/H3 verdicts, Holm correction and leave-one-out external validation do not exist here.
- Experiment B's explore output is complete: 5040 cells / 10 per-sequence JSONL shards; the frozen config and frozen criterion digests are consistent and independently checkable.

## 2. Integrity checks

```bash
# Frozen config: integrity.digest must equal frozen_sha256 in all 5040 cells
python - <<'PY'
import json, glob
cfg = json.load(open("results/noise/frozen_config.json", encoding="utf-8"))
d = cfg["integrity"]["digest"]
rows = [json.loads(l) for p in glob.glob("results/noise/cells_explore/*.jsonl")
        for l in open(p, encoding="utf-8") if l.strip()]
print("cells:", len(rows), "expect 5040")
print("all rows carry frozen hash:", all(r["frozen_sha256"] == d for r in rows))
print("run_explore hash matches  :",
      json.load(open("results/noise/run_explore.json", encoding="utf-8"))["frozen_sha256"] == d)
PY

# Per-file sha256 (POSIX)
sha256sum -c SHA256SUMS.txt

# Equivalent on Windows
python - <<'PY'
import hashlib
bad = []
for line in open("SHA256SUMS.txt", encoding="utf-8"):
    if line.startswith("#") or not line.strip():
        continue
    h, size, path = line.split(None, 2)
    p = path.strip()
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
        bad.append(p)
print("MISMATCH:", bad or "none")
PY
```

Expected: `cells: 5040`, `all rows carry frozen hash: True`, `run_explore hash matches: True`, `MISMATCH: none`.

Key digests:

| File | Digest |
|---|---|
| `results/noise/frozen_config.json` | `71dc864eb6718fac6c0244aac328b964eb608138ce6e92416f516ebb4be76663` (hash of the **config object**, stored in `integrity.digest`) |
| `results/noise/frozen_criterion.json` | `47c32eced24bc80eb3e3790bf2320363d377d46b329f4342468e812f74948af1` |

Note: `integrity.digest` hashes the **config object** (with the digest field removed) as canonical JSON, not the file bytes. Per-file byte hashes are in `SHA256SUMS.txt`. The two serve different purposes; do not conflate them.

## 3. Environment and reproduction path

Original environment: Ubuntu 24.04 / Python 3.12 / ROCm 7.2.1 / torch 2.9.1+rocm / AMD GPU; 15 GB of datasets (MOT17, MOT20, DanceTrack), OSNet weights, TrackEval.

**Not included in this release**: datasets, weights, `.npy` features, SSH private keys, server addresses. Absolute paths in scripts were replaced with `/path/to/...`-style placeholders; change them to your own paths when reproducing (most scripts honour the `MOT_EXP_ROOT` environment variable).

```bash
# 1) Datasets: download directly from a public mirror
bash code/tooling/bootstrap_new_instance.sh <ssh-alias>
bash code/tooling/download_datasets.sh          # run ON the server, 4-way resume

# 2) Phase 1: invert Theta-hat from official det + freeze config + 17/17 self-tests
python code/noise/run_phase1.py --source server --out <out-dir>

# 3) Smoke (one condition, full pipeline)
python code/noise/run_phase2.py --stage smoke --conds 18 --backend gpu --max-frames 200

# 4) Explore stage (5040 cells)
python code/noise/run_phase2.py --stage explore --backend gpu --resume

# 5) Boundaries, fit and freeze criterion P
python code/noise/analyze_explore.py
```

[`code/noise/server_verify.py`](../../code/noise/server_verify.py) recomputes `Θ̂` and compares it with the calibration (field-by-field agreement over 25 sequences). `backup_experiment.py` / `backup_watch.sh` in `code/tooling/` provide incremental mirroring, with results sharded per sequence as JSONL.

## 4. Known limitations (read before reproducing)

- Experiment A's `.npy` features were not retained in the archive, so A's full pipeline **cannot be reproduced bit-for-bit from this repository**; only code, dry-run results and state records remain.
- Experiment B's explore stage **is fully reproducible** (frozen grid + frozen config + 5040 cells + frozen criterion).
- The confirm stage never ran, so no full-scale metrics, Holm correction or leave-one-out validation results exist.
- `background/` contains synthetic E1–E4 as the **motivation** for the two real-benchmark experiments; it is not a third independent experiment.
- All paths and host information were redacted; numbers inside `results/noise/cells_explore/` are measured data.

## 5. Suggested citation wording

If citing this repository, please state explicitly:

> These are archived results from an explore / terminated stage (`final_decision = null`), not a formal verdict on the pre-registered hypotheses; the reported `ΔAssA`, `ΔIDSW` and criterion `P` all come from an explore subset of 10 sequences × 500 frames.
