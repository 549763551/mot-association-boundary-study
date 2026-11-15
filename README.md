# 多目标跟踪中精细关联的收益边界：两次实验的完整归档

**[English version](README.en.md)** · 归档日期 2026-11-15 · 状态：**静态归档，实验已终止**

> **状态声明**
> 本仓库归档两项真基准实验：实验 A 在全量评估完成前终止；实验 B 完成预注册的探索阶段后归档，确认阶段未启动。
> 两项实验的 `final_decision` 均为 `null`，预注册的正式判定流程均未完成。
> 本仓库仅报告已完成的测量、阶段结论与复现材料。

---

## 这个仓库回答什么问题

一个在文献中反复被默认的假设是：当检测器输出存在歧义时，引入更精细的外观关联（ReID、相对差异、一致性门控）能带来可观收益。两项受控实验用于测量该假设的**适用边界**，结果为：

**在 `oracle-detection / association-only` 与受控注入检测噪声这两个口径下，精细关联相对 IoU 基线的边际关联增益几乎为零且统计不显著，却显著增加 ID 切换；而「评测前能否预测这笔收益的符号」这一主研究问题，也不优于简单基线。**

预注册、冻结网格与哈希封装保证了结果的可复现性与可核查性。

## 两次实验

| | 实验 A · 混淆风险触发 + 成对差异仲裁 | 实验 B · 检测噪声梯度 / 关联收益边界 |
|---|---|---|
| 起点 | 合成实验：稀疏差异信号使相对判别比全局判别提前约 `√D` 跨过可分阈值 | 系统查新后，以方法学边界与可预测性为研究对象 |
| 口径 | DanceTrack / MOT17 / MOT20，统一 `oracle-detection`，GT 框按 TrackEval 过滤 | 正交标定 `Θ=(σ_loc, β, λ, τ)`，测 `ΔAssA = AssA(s) − AssA(SORT-IoU)` |
| 策略 | `S0_motion`、`S1_global`、`S2_countgated`、`S3_*` 六种变体 | `A0`–`A5` 六种关联器（含交叉一致性弃权 `A4`） |
| 规模 | 全量目标 36 序列；实际完成 DanceTrack 25/25、MOT17 7/7 特征抽取 | 探索 5040 cells / 840 组 / 84.5 分钟，10 序列 × 500 帧 |
| 归档状态 | 特征抽取阶段之后终止，全量评估与正式判定未完成 | 探索阶段完成并冻结判据；确认阶段未启动 |
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

- **资源投向**：25 个官方 `det.txt` 反解 `Θ̂` 显示，真实检测器的框抖动中位数只有 **0.020**（2% 目标高），而漏检率中位数 **0.432**、连续漏检游程约 **10 帧** → 主要噪声是**时间相关漏检**，不是框位置误差。资源应优先投向检测端的时序恢复与遮挡期轨迹保持，而非继续调框对齐或加精细外观打分。
- **省算力**：`geometry_keyed_v1` —— 只有 `σ_loc` 改变框几何，β/λ/τ 下存活 TP 框与 GT 逐位相同 → 28 个条件里仅 **9 个**需重抽特征。
- **评测实现规范**：官方 HOTA 用 `global_alignment×IoU` 匹配，TP 集合随 tracker id 分配变化；`A1` 丢弃低分框使 `DetA` 不可比；扰动后身份回映射应使用 `src_gid` 旁路通道，而非框坐标键。
- **否定式结论**：「歧义度即可信度」在本口径下不成立，top-1/top-2 边际不宜作为 ReID 门控使用。该结论不提供替代方案。
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

**未包含**：15 GB 数据集、OSNet 权重、抽取的 `.npy` 特征、私钥与服务器凭据。数据集由公开镜像直接下载，见 [复现文档](docs/zh/05-reproduction-and-archive.md)。

## 完整性与可信性

- 冻结配置 [`results/noise/frozen_config.json`](results/noise/frozen_config.json) r4，`integrity.digest = 71dc864e…4be63`，与 5040 个 cell 里逐行写入的 `frozen_sha256` 一致，**全程未改动**。
- 冻结判据 [`results/noise/frozen_criterion.json`](results/noise/frozen_criterion.json) 的 digest 可重跑复现。
- 探索产出完整性门 **11/11 通过**（行数、hash、stage、种子、策略、条件、基准、序列、组数、每组 6 策略、8 个观测量）。
- 跨源校验：MOT17 原始归档与公开镜像重打包数据，25 个序列的 `Θ̂` **逐字段零差异**。
- 每个文件的 sha256 见 [`SHA256SUMS.txt`](SHA256SUMS.txt)。

## 未包含的内容

确认阶段（50 序列 / 50387 帧 / 约 8400 cells / 估计 5–8 小时 GPU）未启动，因此本仓库不包含：正式 H1/H2/H3 判定、Holm 校正、三种留一验证、官方 `det.txt` 锚点双口径、A5 的 RQ3 归因。

## 许可与引用

代码按 MIT 发布；文档与实验数据按 CC BY 4.0 发布（详见 [LICENSE](LICENSE)）。若引用本仓库的负结果或 `Θ̂` 标定，请注明这是**探索 / 终止阶段归档**，而非正式判定。
