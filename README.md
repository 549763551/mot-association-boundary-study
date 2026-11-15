# 多目标跟踪中精细关联的收益边界：两次实验的完整归档

**[English version](README.en.md)** · 归档日期 2026-11-15 · 状态：**已收尾，不再继续**

> **状态声明（请先读）**
> 本仓库是**两次真基准实验的中止/阶段归档**，不是一篇声称取得正向增益的论文。
> 两次实验的 `final_decision` 均为 `null`：
> - 实验 A 因服务器多次断连而中止，全量判定未完成；
> - 实验 B 完成了预注册的探索阶段，用户审核后选择「就此收尾」，确认阶段未启动。
>
> 因此这里的所有结论都是**阶段结论与方向性倾向**，**不是「假设已证伪」**，
> 也不宣称「发现了新方向」或「提出了一个涨点的新方法」。

---

## 这个仓库回答什么问题

一个在文献中反复被默认的假设是：当检测器输出存在歧义时，引入更精细的外观关联（ReID、相对差异、一致性门控）能带来可观收益。我们做了两轮受控实验去测这个假设的**适用边界**，得到的答案是：

**在 `oracle-detection / association-only` 与受控注入检测噪声这两个口径下，精细关联相对 IoU 基线的边际关联增益几乎为零且统计不显著，却显著增加 ID 切换；而「评测前能否预测这笔收益的符号」这一主贡献，也不优于简单基线。**

这不是执行失败。恰恰是预注册 + 冻结网格 + hash 封装让负结果可信、可复现、可引用。

## 两次实验

| | 实验 A · 混淆风险触发 + 成对差异仲裁 | 实验 B · 检测噪声梯度 / 关联收益边界 |
|---|---|---|
| 起点 | 合成实验：稀疏差异信号使相对判别比全局判别提前约 `√D` 跨过可分阈值 | 系统查新后机制新颖被占满，主贡献重定位为方法学 |
| 口径 | DanceTrack / MOT17 / MOT20，统一 `oracle-detection`，GT 框按 TrackEval 过滤 | 正交标定 `Θ=(σ_loc, β, λ, τ)`，测 `ΔAssA = AssA(s) − AssA(SORT-IoU)` |
| 策略 | `S0_motion`、`S1_global`、`S2_countgated`、`S3_*` 六种变体 | `A0`–`A5` 六种关联器（含交叉一致性弃权 `A4`） |
| 规模 | 全量目标 36 序列；实际完成 DanceTrack 25/25、MOT17 7/7 特征抽取 | 探索 5040 cells / 840 组 / 84.5 分钟，10 序列 × 500 帧 |
| 结局 | 三次服务器断连，v3 在 `MOT20-04` 前中断；用户 19:02 决定中止 | 探索完成并冻结判据；用户 22:16 决定收尾，确认阶段未启动 |
| 判定 | `final_decision = null` | `final_decision = null` |

## 关键读数

**实验 B 探索阶段（cluster bootstrap，1000 次，按序列重采样）**

| 量 | 均值 | 95% CI | 读法 |
|---|---|---|---|
| `ΔAssA` (A4−A0) | **+0.001832** | **[−0.000044, +0.004326]** | 含 0 → 平均无显著增益 |
| `ΔIDSW`（负为好） | **+31.43** | **[+25.03, +39.24]** | 显著为正 → 显著多出 ID 切换 |
| `ΔIDF1` | +0.002923 | [+0.000495, +0.005101] | 显著但幅度极小 |

- `ΔAssA` 为正的组占 **49.76%**（≈ 抛硬币）；28 个条件里仅 6 个 CI 不含 0，幅度全 < 0.013。
- `η_flip = 0`：零噪声处 `ΔIDSW` 已是 +0.6，精细关联从无噪声就开始主动制造 ID 切换。
- 判据 `P`（≤8 维无标签逻辑回归，[已冻结](results/noise/frozen_criterion.json)，sha256 `47c32ece…8af1`）：留一序列符号准确率 **0.5512** [0.5190, 0.5857]，**低于**仅置信度基线 0.6071 与 β/λ 基线 0.6048 → 按冻结 `h2_rule`，**H2 倾向不成立**。

**评估口径已实证无误**：自实现 `AssA` / `DetA` 与 TrackEval 官方 `HOTA.eval_sequence` 数值**差值 0**（本机 17/17、服务器 17/17 自检）。

## 这些结果里真正能落地的部分

见 [可落地结论（中）](docs/zh/04-practical-takeaways.md) · [Practical takeaways (EN)](docs/en/04-practical-takeaways.md)。简要说：

- **资源投向**：25 个官方 `det.txt` 反解 `Θ̂` 显示，真实检测器的框抖动中位数只有 **0.020**（2% 目标高），而漏检率中位数 **0.432**、连续漏检游程约 **10 帧** → 主要噪声是**时间相关漏检**，不是框位置误差。该投检测端的时序恢复与遮挡期轨迹保持，而不是继续调框对齐或加精细外观打分。
- **省算力**：`geometry_keyed_v1` —— 只有 `σ_loc` 改变框几何，β/λ/τ 下存活 TP 框与 GT 逐位相同 → 28 个条件里仅 **9 个**需重抽特征。
- **评测别算错**：官方 HOTA 用 `global_alignment×IoU` 匹配，TP 集合随 tracker id 分配变化；A1 丢弃低分框使 DetA 不可比；扰动后不能用框坐标回映射身份（旧口径静默漏 63 行），必须带 `src_gid` 旁路。
- **别做某件事**：「歧义度即可信度」在本口径下不成立，不要拿 top-1/top-2 边际当 ReID 门控上线。这是**否定式**指导，不提供替代方案。
- **不能当阈值用**：`η*_0` / `η†` / `η_flip` 是探索子集上的描述性读数，幅度小、CI 宽、未经留一外部验证与 Holm 校正。

## 文档导航

| 中文 | English |
|---|---|
| [01 实验 A：设计与中止状态](docs/zh/01-original-experiment.md) | [01 Experiment A](docs/en/01-original-experiment.md) |
| [02 实验 B：查新与设计](docs/zh/02-noise-gradient-design.md) | [02 Experiment B: prior art & design](docs/en/02-noise-gradient-design.md) |
| [03 实验 B：探索阶段结果](docs/zh/03-phase2-explore.md) | [03 Experiment B: explore-stage results](docs/en/03-phase2-explore.md) |
| [04 可落地结论与边界](docs/zh/04-practical-takeaways.md) | [04 Practical takeaways](docs/en/04-practical-takeaways.md) |
| [05 复现、完整性与归档](docs/zh/05-reproduction-and-archive.md) | [05 Reproduction & archive](docs/en/05-reproduction-and-archive.md) |

一手材料（中文原件 + 英文对应版）：[`design/`](design/)

## 仓库结构

```
README.md / README.en.md        双版主入口
docs/zh/ · docs/en/             各 5 篇成对文档
design/                         预注册设计稿、阶段报告（中英双版）、分析稿
results/original/               实验 A：预演报告、差异报告、状态记录、图
results/noise/                  实验 B：冻结配置、冻结判据、汇总、标定报告、
                                5040 cells（10 个按序列分片 JSONL）
code/original/ · code/noise/    两次实验的完整可运行代码
code/tooling/                   部署、断点续跑、增量备份脚本
background/                     合成 E1–E4（假设来源，作背景引用，不单列为第三次实验）
SHA256SUMS.txt                  每个发布文件的 sha256
```

**未包含**：15 GB 数据集、OSNet 权重、抽取的 `.npy` 特征、私钥与服务器凭据。数据集走服务器端公开镜像直下，见 [复现文档](docs/zh/05-reproduction-and-archive.md)。

## 完整性与可信性

- 冻结配置 [`results/noise/frozen_config.json`](results/noise/frozen_config.json) r4，`integrity.digest = 71dc864e…4be63`，与 5040 个 cell 里逐行写入的 `frozen_sha256` 一致，**全程未改动**。
- 冻结判据 [`results/noise/frozen_criterion.json`](results/noise/frozen_criterion.json) 的 digest 可重跑复现。
- 探索产出完整性门 **11/11 通过**（行数、hash、stage、种子、策略、条件、基准、序列、组数、每组 6 策略、8 个观测量）。
- 跨源校验：本机原版 `MOT17.zip` 与服务器 hf-mirror 重打包数据，25 个序列的 `Θ̂` **逐字段零差异**。
- 每个文件的 sha256 见 [`SHA256SUMS.txt`](SHA256SUMS.txt)。

## 明确未做的事

按用户决定，**不再**申请服务器补跑全量、不做优化、不做新产物、不换口径、不加消融。确认阶段（50 序列 / 50387 帧 / 约 8400 cells / 估计 5–8 小时 GPU）从未启动，因此以下内容均不存在：正式 H1/H2/H3 判定、Holm 校正、三种留一验证、官方 `det.txt` 锚点双口径、A5 的 RQ3 归因。

## 许可与引用

代码按 MIT 发布；文档与实验数据按 CC BY 4.0 发布（详见 [LICENSE](LICENSE)）。若引用本仓库的负结果或 `Θ̂` 标定，请注明这是**探索/中止阶段归档**，而非正式判定。
