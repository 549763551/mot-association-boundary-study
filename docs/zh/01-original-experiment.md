# 01 · 实验 A：混淆风险触发的成对差异仲裁（已中止）

[返回主入口](../../README.md) · [English](../en/01-original-experiment.md)

> **状态**：中止。全量判定未完成，`final_decision = null`。**不是「已证伪」**。

## 1. 假设从哪里来

合成实验（[`background/`](../../background/) 中的 E1–E4）发现一个可复现的现象：

- 当两类样本的差异信号**稀疏**时，成对/相对判别比全局判别提前约 `√D` 跨过可分性阈值；
- 因此存在一段「相对判别已可分、全局判别尚不可分」的窗口，在这段窗口里引入成对差异仲裁应当有正收益；
- 收益是**倒 U 形**的：噪声太低时全局判别已经够用（仲裁只是成本），噪声太高时两者都不可分。

由此预注册四条待验证结论：

1. 收益随混淆强度呈倒 U 形；
2. 临界带出现历史置信度凹口；
3. 触发成本会在高触发率处反超收益；
4. 在线模板更新会使仲裁退化。

## 2. 真基准实验设计

| 项 | 设定 |
|---|---|
| 数据集 | DanceTrack（val 25 序列）、MOT17（train 7 序列）、MOT20（train 4 序列） |
| 统一口径 | `oracle-detection / association-only`：用 GT 框作为检测输入，只改变关联策略 |
| GT 过滤 | 按 TrackEval 的 `MotChallenge2DBox` 口径：保留 `conf != 0` 且 `class == 1`（行人） |
| 特征 | OSNet（MSMT17 / ImageNet 权重），FP32，跨帧攒批，ROCm GPU |
| 关联器 | `S0_motion`（IoU + 线性速度）、`S1_global`（全局外观）、`S2_countgated`（计数门控）、`S3_frame_calib` / `S3_hist_auc` / `S3_online` / `S3_saturated` / `S3_no_abstain` |
| 评估 | TrackEval `HOTA / CLEAR / Identity`；官方汇总读 `pedestrian_detailed.csv` 的 `COMBINED` 行 |
| 统计 | `TAU_GRID` × `ALPHA_GRID` × 多种子，bootstrap 95% CI |

**设计目的**是把检测器因素彻底隔离，单独看「混淆风险触发 + 成对差异仲裁」这一机制本身能否带来关联增益。代价是 `DetA ≈ 0.97`，结论**不能外推到真实检测器输出**——这个局限后来直接催生了实验 B。

## 3. 实际完成的证据（全部为非正式信号）

**全量远程流程**：DanceTrack 25/25、MOT17 7/7 序列的 OSNet 特征抽取完成；在 `MOT20-04` 特征抽取前，远程实例第三次断开（`V3_SCRIPT_EXIT=1`）。诊断、跟踪、官方评估、分析、差异报告均未跑完。

**本机 CPU 预演**（5 序列 × 250 帧，[`results/original/full_experiment_report.json`](../../results/original/full_experiment_report.json)）：

- `all_conclusions_reproduced = False`；
- `S0_motion` 最好，**所有 `S3` 变体都更差**；
- 四条预注册结论中仅「触发成本反超」成立，其余三条不成立或证据不足；
- `alpha = 1.0`（全量触发）时 IDSW 达 916。

**oracle 口径下的非正式读数**：`S3_frame_calib` 触发率约 1.2%，饱和配置 100%；DanceTrack 特征 AUC 0.6807、MOT17 0.8875。

这些数的方向一致地指向「机制在该口径下无正收益」，但**样本是子集、不是全量，且没有完成预注册的正式判定流程**，所以只能作为倾向记录。

## 4. 统计口径修复（重要，影响结论可信度）

预演前修掉一个真实的统计 bug：`analyze.py` 里计算逐帧 ID 切换时用了 `prev = cur`，会在遮挡后漏计切换。改为：

```python
for gid, tid in cur.items():
    if gid in prev and prev[gid] != tid:
        switches += 1
prev.update(cur)
```

修完加入 `idsw_consistency` 质量门，本机预演自实现 IDSW `15` 对 TrackEval 官方 `15` 一致。这个教训在实验 B 里被再次验证——注入扰动后，`_key(frame, round(box,1))` 的框坐标身份回映射会**静默漏掉 63 个真实目标行**，因此实验 B 改用 `src_gid` 旁路通道。

## 5. 中止原因与最终状态

远程实例先后三次被回收/断连（端口几经更换），特征 `.npy` 在一次断连中未同步到本机。用户于 2026-11-15 19:02 明确决定**停止该实验、转到新方向**，不再申请服务器补跑全量。

因此：

- `final_decision` 保持 `null`；
- 四条预注册结论**既未证实也未证伪**；
- 没有做优化、没有做新产物、没有换口径；
- 本实验的结论不得写成「合成假设在真基准上被证伪」。

## 6. 复现材料

- 代码：[`code/original/`](../../code/original/)（`exp_config.py`、`data_utils.py`、`extract_features.py`、`tracker.py`、`evaluate.py`、`analyze.py`、`make_diff_report.py` 等）
- 结果：[`results/original/`](../../results/original/)（预演报告、差异报告、状态记录 `partial_results.json`、图）
- 背景：[`background/`](../../background/)（合成 E1–E4 与三张图）
