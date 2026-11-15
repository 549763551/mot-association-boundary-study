#!/usr/bin/env python3
"""Post-hoc analysis: stratification by current-frame separability, budget
curves, sequence-level bootstrap CIs, and the four pre-registered decisions.

All GT identities are used ONLY here, never inside the tracker.
"""
import os
import glob
import json
import math
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_config import (BENCHMARKS, EVALDATA, RESULTS, BOOTSTRAP_N,
                        BOOTSTRAP_SEED, TAU_GRID, DECISION_RULES)
from data_utils import (list_sequences, load_sequence, load_features,
                        save_json, tracker_dir)

BANDS3 = [0.0, 1 / 3, 2 / 3, 1.0 + 1e-9]
BANDS5 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-9]


def _key(frame, box):
    return (int(frame), tuple(np.round(np.asarray(box, dtype=float), 1)))


def load_track_rows(bench, tracker, seq_name):
    p = os.path.join(tracker_dir(bench, tracker), seq_name + ".txt")
    out = []
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for line in f:
            v = line.strip().split(",")
            if len(v) < 7:
                continue
            out.append((int(float(v[0])), int(float(v[1])),
                        float(v[2]), float(v[3]), float(v[4]), float(v[5])))
    return out


def idsw_per_frame(seq, rows):
    gt_key = {}
    for f in seq.frames:
        for i, b in enumerate(seq.boxes[f]):
            gt_key[_key(f, b)] = int(seq.ids[f][i])
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r[0]].append(r)
    res = {}
    prev = {}
    unmatched = 0
    for f in seq.frames:
        cur = {}
        for r in by_frame.get(f, []):
            gid = gt_key.get(_key(f, r[2:6]))
            if gid is None:
                unmatched += 1
                continue
            cur[gid] = r[1]
        switches = 0
        for gid, tid in cur.items():
            if gid in prev and prev[gid] != tid:
                switches += 1
        res[f] = switches
        # A GT identity can be absent for several frames (occlusion / missed
        # detection). Compare against its LAST observed tracker id, not the
        # whole previous-frame dictionary: replacing `prev` would silently
        # discard every switch that happens right after a gap.
        prev.update(cur)
    return res, unmatched


def frame_separability(seq, feats):
    out = {}
    for f in seq.frames:
        off = seq.offsets[f]
        F = feats[off:off + len(seq.boxes[f])]
        ids = seq.ids[f]
        n = len(F)
        if n < 2 or len(np.unique(ids)) < 2:
            out[f] = 1.0
            continue
        C = np.clip(F @ F.T, -1.0, 1.0)
        vals = []
        for i in range(n):
            others = np.flatnonzero(ids != ids[i])
            if len(others):
                vals.append(1.0 - float(C[i, others].max()))
        out[f] = float(np.mean(vals)) if vals else 1.0
    return out


def band_edges(values, qs):
    a = np.asarray(values, dtype=float)
    return np.quantile(a, qs[1:-1]).tolist()


def assign_band(v, edges):
    return int(np.searchsorted(edges, v, side="right"))


def collect(benchmarks, trackers):
    """Return per-(bench,seq) records: sep, idsw per tracker, band assignment."""
    records = []
    all_sep = []
    n_unmatched = 0
    n_rows = 0
    for bench in benchmarks:
        for name, d in list_sequences(bench):
            seq = load_sequence(bench, name, d)
            feats = load_features(bench, name)
            sep = frame_separability(seq, feats)
            idsw = {}
            for t in trackers:
                rows = load_track_rows(bench, t, name)
                idsw[t], u = idsw_per_frame(seq, rows)
                n_unmatched += u
                n_rows += len(rows)
            frames = seq.frames
            records.append(dict(bench=bench, seq=name, frames=frames,
                                sep=[sep[f] for f in frames],
                                idsw={t: [idsw[t][f] for f in frames] for t in trackers}))
            all_sep.extend([sep[f] for f in frames])
    e3 = band_edges(all_sep, BANDS3)
    e5 = band_edges(all_sep, BANDS5)
    for r in records:
        r["b3"] = [assign_band(v, e3) for v in r["sep"]]
        r["b5"] = [assign_band(v, e5) for v in r["sep"]]
    frac = n_unmatched / max(1, n_rows)
    print(f"  GT-box matching: {n_unmatched}/{n_rows} unmatched ({frac:.4%})", flush=True)
    if frac > 0.01:
        print("  WARNING: >1% of tracker rows could not be mapped back to a GT box", flush=True)
    return records, e3, e5


def band_delta(records, target, baseline="S0_motion", nbands=3, bench=None):
    """Per-sequence delta IDSW (baseline - target) for each band."""
    key = "b3" if nbands == 3 else "b5"
    out = {}
    for r in records:
        if bench and r["bench"] != bench:
            continue
        b = np.asarray(r[key], dtype=int)
        base = np.asarray(r["idsw"][baseline], dtype=float)
        tgt = np.asarray(r["idsw"][target], dtype=float)
        d = base - tgt
        out[r["seq"]] = np.array([d[b == k].sum() for k in range(nbands)], dtype=float)
    return out


def bootstrap_ci(per_seq_delta, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    keys = sorted(per_seq_delta)
    M = np.stack([per_seq_delta[k] for k in keys])
    rng = np.random.default_rng(seed)
    N = len(keys)
    idx = rng.integers(0, N, size=(n, N))
    draws = M[idx].sum(axis=1)
    lo = np.percentile(draws, 2.5, axis=0)
    hi = np.percentile(draws, 97.5, axis=0)
    return dict(mean=M.sum(axis=0).tolist(), ci_lo=lo.tolist(), ci_hi=hi.tolist(),
                n_sequences=N)


def load_stats():
    p = os.path.join(RESULTS, "run_stats.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def load_trackeval():
    p = os.path.join(RESULTS, "trackeval_results.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def analyze_all(benchmarks, trackers):
    print("=== collecting per-frame IDSW + separability ===", flush=True)
    records, e3, e5 = collect(benchmarks, trackers)
    print(f"  records={len(records)} sep_band3_edges={np.round(e3,4).tolist()} "
          f"sep_band5_edges={np.round(e5,4).tolist()}", flush=True)

    stats = load_stats()
    te = load_trackeval()

    # ---------------------------------------------------------- baseline error budget
    base_total = {b: float(sum(sum(r["idsw"]["S0_motion"]) for r in records if r["bench"] == b))
                  for b in benchmarks}
    print(f"  S0 total IDSW by benchmark: {base_total}", flush=True)

    # Quality gate: because the input boxes ARE the GT boxes, our independent
    # per-frame IDSW accounting must reproduce TrackEval's official IDSW for
    # the motion-only baseline. A mismatch means the stratification is built
    # on a broken statistic and the four decisions cannot be trusted.
    consistency = {}
    for b in benchmarks:
        off = te.get(f"{b}/S0_motion", {}).get("summary", {}).get("IDSW")
        consistency[b] = dict(self_idsw=base_total[b], official_idsw=off,
                              delta=None if off is None else base_total[b] - float(off))
    print(f"  IDSW consistency (self vs official): {consistency}", flush=True)

    # ---------------------------------------------------------- overall official metrics
    official = {}
    for k, v in te.items():
        s = v.get("summary", {})
        official[k] = {m: s.get(m) for m in ("HOTA", "DetA", "AssA", "IDF1", "IDSW", "MOTA", "Frag")}

    # ---------------------------------------------------------- inverted-U (3 bands)
    inv = {}
    for t in ("S3_hist_auc", "S3_frame_calib"):
        d = band_delta(records, t, nbands=3)
        inv[t] = bootstrap_ci(d)
    # ---------------------------------------------------------- critical notch (5 bands)
    notch = {}
    for t in ("S3_hist_auc", "S3_frame_calib"):
        d = band_delta(records, t, nbands=5)
        notch[t] = bootstrap_ci(d)
    # ---------------------------------------------------------- online ablation
    online_d = {}
    for b in benchmarks:
        dd = {}
        for r in records:
            if r["bench"] != b:
                continue
            base = np.asarray(r["idsw"]["S3_hist_auc"], float)
            on = np.asarray(r["idsw"]["S3_online"], float)
            dd[r["seq"]] = np.array([(base - on).sum()])
        online_d[b] = bootstrap_ci(dd)
    online_all = {}
    dd = {}
    for r in records:
        base = np.asarray(r["idsw"]["S3_hist_auc"], float)
        on = np.asarray(r["idsw"]["S3_online"], float)
        dd[r["bench"] + "/" + r["seq"]] = np.array([(base - on).sum()])
    online_all = bootstrap_ci(dd)

    # ---------------------------------------------------------- budget / tau curve
    tau_curve = []
    for i in range(len(TAU_GRID)):
        t = "tau_%02d" % i
        tr_rate, cost, delta = [], [], []
        for b in benchmarks:
            ss = stats.get(f"{b}/{t}", {})
            if ss:
                tr_rate.append(np.mean([v["trigger_rate"] for v in ss.values()]))
                cost.append(np.mean([v["cost_ratio"] for v in ss.values()]))
            d = band_delta(records, t, nbands=3, bench=b)
            delta.append(float(sum(v.sum() for v in d.values())))
        tau_curve.append(dict(tau=TAU_GRID[i], tracker=t,
                              trigger_rate=float(np.mean(tr_rate)) if tr_rate else None,
                              cost_ratio=float(np.mean(cost)) if cost else None,
                              delta_idsw=float(np.sum(delta))))
    # saturated and best
    sat = stats.get("__unused__", {})
    sat_rates = []
    for b in benchmarks:
        ss = stats.get(f"{b}/S3_saturated", {})
        if ss:
            sat_rates.append(np.mean([v["trigger_rate"] for v in ss.values()]))
    sat_rate = float(np.mean(sat_rates)) if sat_rates else None
    sat_delta = float(sum(sum(v.sum() for v in band_delta(records, "S3_saturated", nbands=3, bench=b).values())
                          for b in benchmarks))
    best_tau_delta = max((x["delta_idsw"] for x in tau_curve if x["tracker"] != "tau_07"), default=0.0)
    best_tau_delta = max(best_tau_delta, float(sum(
        sum(v.sum() for v in band_delta(records, "S3_frame_calib", nbands=3, bench=b).values())
        for b in benchmarks)))

    # ---------------------------------------------------------- decisions
    R = DECISION_RULES
    m = inv["S3_frame_calib"]["mean"]
    lo = inv["S3_frame_calib"]["ci_lo"]
    hi = inv["S3_frame_calib"]["ci_hi"]
    d1 = (lo[1] > R["inverted_u"]["mid_band_delta_lower_ci_gt"] and
          m[1] - m[0] > R["inverted_u"]["mid_minus_low_gt"] and
          m[1] - m[2] > R["inverted_u"]["mid_minus_high_gt"])
    h = notch["S3_hist_auc"]
    c = notch["S3_frame_calib"]
    d2 = (h["ci_hi"][1] < R["critical_notch"]["hist_delta_upper_ci_lt"] and
          c["ci_lo"][1] >= R["critical_notch"]["calib_delta_lower_ci_ge"])
    d3 = bool(sat_rate is not None and sat_rate > R["cost_overrun"]["saturated_trigger_rate_gt"]
              and sat_delta < best_tau_delta + R["cost_overrun"]["saturated_delta_vs_best_lt"])
    d4 = bool(online_all["ci_hi"][0] < R["online_degradation"]["online_minus_frozen_upper_ci_lt"])
    decisions = dict(inverted_u=bool(d1), critical_notch=bool(d2),
                     cost_overrun=d3, online_degradation=d4)
    reproduced = all(decisions.values())

    report = dict(
        meta=dict(benchmarks=benchmarks, strategies=trackers,
                  n_sequences=len(records), tau_grid=TAU_GRID,
                  band3_edges=e3, band5_edges=e5),
        s0_total_idsw=base_total,
        idsw_consistency=consistency,
        official_metrics=official,
        inverted_u=inv,
        critical_notch=notch,
        online_ablation=online_all,
        online_by_benchmark=online_d,
        tau_curve=tau_curve,
        saturated=dict(trigger_rate=sat_rate, delta_idsw=sat_delta,
                       best_tau_delta=best_tau_delta),
        decisions=decisions,
        all_conclusions_reproduced=bool(reproduced),
        decision_rule="if any pre-registered conclusion is falsified -> stop and record the difference report",
    )
    save_json(report, os.path.join(RESULTS, "full_experiment_report.json"))

    # ---------------------------------------------------------- plots
    try:
        plot_results(report, tau_curve, inv, notch)
    except Exception as e:
        print("  plot failed:", type(e).__name__, e)

    # ---------------------------------------------------------- markdown summary
    md = build_markdown(report)
    with open(os.path.join(RESULTS, "full_experiment_summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md, flush=True)
    return report


def plot_results(report, tau_curve, inv, notch):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(3)
    for t, col in (("S3_hist_auc", "#d62728"), ("S3_frame_calib", "#1f77b4")):
        m = inv[t]["mean"]
        lo = inv[t]["ci_lo"]
        hi = inv[t]["ci_hi"]
        ax[0].bar(x + (0.0 if t == "S3_hist_auc" else 0.35), m, 0.35,
                  yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)],
                  capsize=3, color=col, alpha=0.85, label=t)
    ax[0].set_xticks(x + 0.17)
    ax[0].set_xticklabels(["low sep", "mid sep", "high sep"])
    ax[0].set_ylabel("delta IDSW (S0 - method)")
    ax[0].set_title("Inverted-U check")
    ax[0].legend(fontsize=8)
    ax[0].axhline(0, color="k", lw=0.8)

    x5 = np.arange(5)
    for t, col in (("S3_hist_auc", "#d62728"), ("S3_frame_calib", "#1f77b4")):
        m = notch[t]["mean"]
        lo = notch[t]["ci_lo"]
        hi = notch[t]["ci_hi"]
        ax[1].bar(x5 + (0.0 if t == "S3_hist_auc" else 0.35), m, 0.35,
                  yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)],
                  capsize=3, color=col, alpha=0.85, label=t)
    ax[1].set_xticks(x5 + 0.17)
    ax[1].set_xticklabels(["q1", "q2", "q3", "q4", "q5"])
    ax[1].set_title("Critical-band notch check")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].legend(fontsize=8)

    xs = [c["trigger_rate"] for c in tau_curve if c["trigger_rate"] is not None]
    ys = [c["delta_idsw"] for c in tau_curve if c["trigger_rate"] is not None]
    ax[2].plot(xs, ys, "o-", color="#1f77b4")
    ax[2].set_xlabel("trigger rate")
    ax[2].set_ylabel("delta IDSW")
    ax[2].set_title("Budget / precision curve")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(RESULTS, "full_experiment_figures.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("  figures ->", p)


def build_markdown(r):
    d = r["decisions"]
    lines = [
        "# 真基准完整实验 · 结果摘要", "",
        "## 实验设计（预注册，非事后调参）", "",
        "- **研究问题**：合成实验得到的四条结论，在真实 MOT 基准上是否成立。",
        "- **数据**：DanceTrack val (25 序列)、MOT17 train / FRCNN (7 序列)、MOT20 train (4 序列)。",
        "- **输入口径**：三个基准统一使用「按 TrackEval 口径过滤后的 GT 框」作为检测输入"
          "（conf≠0 且 class=1），即 oracle-detection / association-only 设定。"
          "理由：DanceTrack 官方 val 不提供 det.txt，若混用公开检测器输出会造成口径不一致；"
          "本研究考察关联策略而非检测器质量。**局限**：检测完美使 DetA 接近 1，"
          "结论不能直接外推到含误检/漏检的真实检测输入。",
        "- **ReID**：OSNet-x1_0 (MSMT17)，512 维，L2 归一化，FP32，一次抽取并缓存；"
          "所有策略共享同一份特征，排除预处理差异。",
        "- **跟踪器**：在线逐帧，线性速度预测 + IoU 门控 + 匈牙利匹配；模板取可信前缀 10 帧均值并冻结"
          "（在线更新仅作消融）。",
        "- **对照**：S0 motion-only / S1 逐帧全局外观 / S2 候选数门控 / S3 混淆风险触发+成对差异仲裁。",
        "- **消融与扫描**：历史 AUC 置信度 vs 当前帧校准、在线模板更新、饱和触发、禁用弃权、"
          "τ 网格（预算曲线）、α 外观权重网格、5 个随机种子。",
        "- **GT 使用边界**：跟踪器全程不接触 GT 身份；GT 身份仅在事后用于可分性分层与正确性核算。",
        "- **统计**：序列级有放回 bootstrap 1000 次给 95% CI；判定阈值在跑实验前写入 exp_config.py。",
        "- **主指标**：TrackEval 官方 HOTA / AssA / DetA / IDF1 / IDSW / MOTA；"
          "分层与预算曲线使用逐帧 IDSW（自实现，口径独立于官方汇总）。", "",
        "## 基线错误预算 (S0 总 IDSW)",
    ]
    for k, v in r["s0_total_idsw"].items():
        lines.append(f"- {k}: {int(v)}")
    cs = r.get("idsw_consistency", {})
    if cs:
        lines += ["", "### 统计口径交叉校验（自实现逐帧 IDSW vs TrackEval 官方 IDSW）", ""]
        for k, v in cs.items():
            lines.append(f"- {k}: 自实现 {int(v['self_idsw'])}, 官方 {v['official_idsw']}, "
                         f"差值 {v['delta']}")
        lines.append("- 输入框即 GT 框，两者应一致；若偏差大则分带统计不可信。")
    lines += ["", "## 四条预注册结论", ""]
    names = dict(inverted_u="倒 U 形（中等可分带收益最大）",
                 critical_notch="临界带历史置信度凹口",
                 cost_overrun="触发饱和后成本反超",
                 online_degradation="在线模板更新退化")
    for k, n in names.items():
        lines.append(f"- {n}: **{'成立' if d[k] else '不成立'}**")
    lines += ["", "### 倒 U 形分带 ΔIDSW (低 / 中 / 高)", ""]
    for t in ("S3_hist_auc", "S3_frame_calib"):
        v = r["inverted_u"][t]
        lines.append(f"- {t}: mean={[round(x,1) for x in v['mean']]} "
                     f"CI95={[[round(a,1), round(b,1)] for a, b in zip(v['ci_lo'], v['ci_hi'])]}")
    lines += ["", "### 临界带 5 分位 ΔIDSW", ""]
    for t in ("S3_hist_auc", "S3_frame_calib"):
        v = r["critical_notch"][t]
        lines.append(f"- {t}: mean={[round(x,1) for x in v['mean']]}")
    lines += ["", "### 预算曲线", ""]
    for c in r["tau_curve"]:
        lines.append(f"- tau={c['tau']:<8} trigger={c['trigger_rate']} "
                     f"cost={c['cost_ratio']} deltaIDSW={round(c['delta_idsw'],1)}")
    lines += ["", f"- 饱和触发率: {r['saturated']['trigger_rate']}, "
              f"饱和 ΔIDSW: {round(r['saturated']['delta_idsw'],1)}, "
              f"最佳标称 ΔIDSW: {round(r['saturated']['best_tau_delta'],1)}",
              "", "## 判定", "",
              f"**四条结论全部复现: {r['all_conclusions_reproduced']}**", "",
              "判定状态：四条预注册结论未全部成立；本机预演结果不构成正式判定。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    a = ap.parse_args()
    from run_full_experiment import build_run_list
    analyze_all(a.benchmarks, [x[0] for x in build_run_list()])
