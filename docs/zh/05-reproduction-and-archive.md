# 05 · 复现、完整性与归档状态

[返回主入口](../../README.md) · [English](../en/05-reproduction-and-archive.md)

## 1. 归档状态

- 实验 A 在全量评估完成前终止；实验 B 的确认阶段未启动。两项实验的 `final_decision` 均为 `null`。
- 本仓库为静态归档，材料定位为「探索阶段结论 + 方向性倾向」；H1/H2/H3 的正式判定、Holm 校正与留一外部验证均不存在。
- 实验 B 的探索产出完整：5040 cells / 10 个按序列分片 JSONL；冻结配置与冻结判据的摘要一致且可复核。

## 2. 完整性校验

```bash
# 冻结配置：integrity.digest 与 5040 个 cell 的 frozen_sha256 必须一致
python - <<'PY'
import json, glob, hashlib
cfg = json.load(open("results/noise/frozen_config.json", encoding="utf-8"))
d = cfg["integrity"]["digest"]
rows = [json.loads(l) for p in glob.glob("results/noise/cells_explore/*.jsonl")
        for l in open(p, encoding="utf-8") if l.strip()]
print("cells:", len(rows), "expect 5040")
print("all rows carry frozen hash:", all(r["frozen_sha256"] == d for r in rows))
print("run_explore hash matches  :",
      json.load(open("results/noise/run_explore.json", encoding="utf-8"))["frozen_sha256"] == d)
PY

# 每个发布文件的 sha256
sha256sum -c SHA256SUMS.txt 2>/dev/null || python - <<'PY'
# Windows 无 sha256sum 时的等价校验
import hashlib, os
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

预期：`cells: 5040`、`all rows carry frozen hash: True`、`run_explore hash matches: True`、`MISMATCH: none`。

其余关键 hash：

| 文件 | digest |
|---|---|
| `results/noise/frozen_config.json` | `71dc864eb6718fac6c0244aac328b964eb608138ce6e92416f516ebb4be76663`（配置对象 hash，写入 `integrity.digest`） |
| `results/noise/frozen_criterion.json` | `47c32eced24bc80eb3e3790bf2320363d377d46b329f4342468e812f74948af1` |

注意：`integrity.digest` 是对**配置对象**（剔除 digest 字段后）的规范化 JSON hash，不是文件字节的 sha256；逐文件字节 hash 见 `SHA256SUMS.txt`。两者用途不同，不要混用。

## 3. 环境与复现路径

原始运行环境：Ubuntu 24.04 / Python 3.12 / ROCm 7.2.1 / torch 2.9.1+rocm / AMD GPU；数据集 15 GB（MOT17、MOT20、DanceTrack），OSNet 权重，TrackEval。

**发布包不包含**数据集、权重、`.npy` 特征、SSH 私钥或服务器地址。脚本里的绝对路径已替换为 `/path/to/...` 形式的占位符，复现时请改成自己的路径（多数脚本支持 `MOT_EXP_ROOT` 环境变量覆盖）。

```bash
# 1) 准备数据集：从公开镜像直接下载
bash code/tooling/bootstrap_new_instance.sh <ssh-alias>
bash code/tooling/download_datasets.sh          # 服务器端执行，4 路断点续传

# 2) Phase 1：官方 det 反解 Θ̂ + 冻结配置 + 17/17 自检
python code/noise/run_phase1.py --source server --out <out-dir>

# 3) 冒烟（单条件，验证全链路）
python code/noise/run_phase2.py --stage smoke --conds 18 --backend gpu --max-frames 200

# 4) 探索阶段（5040 cells）
python code/noise/run_phase2.py --stage explore --backend gpu --resume

# 5) 汇总边界量、拟合并冻结判据 P
python code/noise/analyze_explore.py
```

跨源校验脚本 [`code/noise/server_verify.py`](../../code/noise/server_verify.py) 会重算 `Θ̂` 并与标定结果比对（25 序列逐字段零差异）。`code/tooling/` 中的 `backup_experiment.py` / `backup_watch.sh` 提供增量镜像，结果按序列分片 JSONL 存储。

## 4. 已知限制（复现前必读）

- 实验 A 的抽取 `.npy` 特征未随归档保留，因此 A 的全量流程**无法从本仓库逐位复现**；只保留了代码、预演结果与状态记录。
- 实验 B 的探索阶段**可完整复现**（冻结网格 + 冻结配置 + 5040 cells + 冻结判据）。
- 确认阶段从未运行，因此不存在全量指标、Holm 校正或留一验证结果。
- `background/` 中的合成 E1–E4 是两次真基准实验的**动机来源**，不作为第三次独立实验。
- 所有路径与主机信息均已脱敏；`results/noise/cells_explore/` 中的数值均为测量数据。

## 5. 引用建议

若引用本仓库，建议明确写出：

> 该结果为探索 / 终止阶段的归档（`final_decision = null`），不是对预注册假设的正式判定；所报告的 `ΔAssA` / `ΔIDSW` / 判据 `P` 均来自 10 序列 × 500 帧的探索子集。
