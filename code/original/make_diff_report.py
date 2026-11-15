#!/usr/bin/env python3
"""Generate the synthetic-vs-real difference report for user review."""
import os
import json

from exp_config import RESULTS


def load(name, default=None):
    p = name if os.path.isabs(name) else os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return default if default is not None else {}
    return json.load(open(p))


def synth_summary(s):
    e2 = s.get("e2", {})
    rows = e2.get("by_delta_cast", [])
    per_delta = {}
    for r in rows:
        if r.get("cast"):
            continue
        de = r["delta"]
        per_delta.setdefault(de, {})[r["policy"]] = r
    lines = ["| δ | motion IDF1 | global IDF1 | confusion IDF1 | Δ(conf-motion) | confusion calls |",
             "|---|---|---|---|---|---|"]
    for de in sorted(per_delta):
        p = per_delta[de]
        m, g, c = p["motion"], p["global"], p["confusion"]
        lines.append(f"| {de:.2f} | {m['IDF1']:.4f} | {g['IDF1']:.4f} | {c['IDF1']:.4f} | "
                     f"{c['IDF1']-m['IDF1']:+.4f} | {c['calls']:.1f} |")
    e1 = s.get("e1", {}).get("by_delta_cast", [])
    e1map = {r["delta"]: r for r in e1 if not r.get("cast")}
    lines += ["", "| δ | global acc | diff acc | sep AUC | 领先幅度 |",
              "|---|---|---|---|---|"]
    for de in sorted(e1map):
        r = e1map[de]
        ga, da = r["global_acc"], r["diff_acc"]
        lead = "n/a" if da is None else f"{da-ga:+.4f}"
        lines.append(f"| {de:.2f} | {ga:.4f} | {'-' if da is None else f'{da:.4f}'} | "
                     f"{r['sep_auc']:.4f} | {lead} |")
    e3 = s.get("e3", [])
    lines += ["", "| 消融 | IDF1 | IDSW | calls |", "|---|---|---|---|"]
    for r in e3:
        lines.append(f"| {r['ablation']} | {r['IDF1']:.4f} | {r['IDSW']:.3f} | {r['calls']:.1f} |")
    return "\n".join(lines)


def real_official_table(official):
    rows = ["| benchmark / tracker | HOTA | AssA | DetA | IDF1 | IDSW | MOTA |",
            "|---|---|---|---|---|---|---|"]
    for k in sorted(official):
        v = official[k]
        fmt = lambda x: "-" if x is None else f"{x:.4f}" if isinstance(x, float) and abs(x) < 2 else f"{x:.0f}"
        rows.append(f"| {k} | {fmt(v.get('HOTA'))} | {fmt(v.get('AssA'))} | {fmt(v.get('DetA'))} | "
                    f"{fmt(v.get('IDF1'))} | {fmt(v.get('IDSW'))} | {fmt(v.get('MOTA'))} |")
    return "\n".join(rows)


def main():
    r = load("full_experiment_report.json")
    diag = load("feature_diagnostics.json")
    syn_path = os.path.join(os.environ.get("MOT_EXP_ROOT", "/root/mot_exp"),
                            "exp", "synthetic_experiment.json")
    syn = load(syn_path)
    if not r:
        print("NO_REAL_REPORT")
        return
    d = r["decisions"]
    reproduced = r["all_conclusions_reproduced"]
    names = dict(inverted_u="倒 U 形：中等可分带收益最大",
                 critical_notch="临界带历史置信度失配凹口",
                 cost_overrun="触发饱和后成本反超",
                 online_degradation="在线模板更新退化")

    L = []
    L += ["# 合成假设 → 真基准：结论比对报告", "",
          f"**判定：四条结论全部复现 = {reproduced}**", ""]
    if reproduced:
        L += ["按约定：结论一致，进入优化 / 新产物阶段。", ""]
    else:
        L += ["按约定：**结论在真数据上被推翻，实验到此停止，本文件提交审核，暂不做优化或新产物。**", ""]
    L += ["| 预注册结论 | 真数据判定 |", "|---|---|"]
    for k, n in names.items():
        L.append(f"| {n} | {'成立' if d.get(k) else '不成立'} |")
    L += ["", "## 1. 合成实验原始基线（对照数字）", "",
          "e2 各可分性带（cast=false，n=5/格）：", "", synth_summary(syn), "",
          "关键机制证据：", "",
          "- e1：δ=0.10 时相对判别 acc 0.9684 显著领先全局 0.7461；δ≥0.28 时两者同为 1.0，"
          "即「全局判别追上后无错可修」，收益窗口只在两条曲线错开处。",
          "- e3：在线模板更新使 IDF1 从完整方法 0.9256 跌到 0.8288，calls 从 22.7 涨到 199.6。",
          "- e2 overall：confusion calls=22.73，global calls=480，成本约 4.7%。", "",
          "## 2. 真基准官方指标（TrackEval）", "", real_official_table(r.get("official_metrics", {})), "",
          "## 3. 基线错误预算与特征前提诊断", ""]
    for k, v in r.get("s0_total_idsw", {}).items():
        dm = diag.get(k, {})
        L.append(f"- {k}: S0 总 IDSW = {int(v)}；逐对可分性 AUC = "
                 f"{dm.get('auc', float('nan')):.4f}；同 ID 余弦 {dm.get('pos_mean', float('nan')):.4f}；"
                 f"异 ID 余弦 {dm.get('neg_mean', float('nan')):.4f}；"
                 f"top64 差异能量 {dm.get('sparse_energy', float('nan')):.4f}")
    L += ["", "## 4. 分带收益（ΔIDSW = S0 − 方法，正值为收益；95% CI 来自序列级 bootstrap）", ""]
    for t in ("S3_hist_auc", "S3_frame_calib"):
        v = r["inverted_u"][t]
        L.append(f"- {t}（低/中/高）: mean={[round(x, 1) for x in v['mean']]} "
                 f"CI95={[[round(a, 1), round(b, 1)] for a, b in zip(v['ci_lo'], v['ci_hi'])]}")
    L += ["", "5 分位临界带检查：", ""]
    for t in ("S3_hist_auc", "S3_frame_calib"):
        v = r["critical_notch"][t]
        L.append(f"- {t}: mean={[round(x, 1) for x in v['mean']]} "
                 f"CI95={[[round(a, 1), round(b, 1)] for a, b in zip(v['ci_lo'], v['ci_hi'])]}")
    L += ["", "## 5. 预算 / 触发曲线", "",
          "| τ | 触发率 | 成本比 | ΔIDSW |", "|---|---|---|---|"]
    for c in r.get("tau_curve", []):
        L.append(f"| {c['tau']} | {c['trigger_rate']} | {c['cost_ratio']} | {round(c['delta_idsw'],1)} |")
    sat = r.get("saturated", {})
    L += ["", f"- 饱和配置：触发率 {sat.get('trigger_rate')}，ΔIDSW {sat.get('delta_idsw')}；"
          f"最佳标称 ΔIDSW {sat.get('best_tau_delta')}", "",
          "## 6. 在线模板更新消融", "",
          f"- frozen − online 的 ΔIDSW = {r['online_ablation']['mean']}，"
          f"CI95 = [{r['online_ablation']['ci_lo']}, {r['online_ablation']['ci_hi']}]", "",
          "## 7. 判定规则（跑实验前固定）", "",
          "```json", json.dumps(r.get("meta", {}), ensure_ascii=False, indent=2)[:1200], "```", "",
          "## 8. 实验设定与局限", "",
          "- 三基准统一使用过滤后 GT 框作为检测输入（oracle-detection / association-only），"
          "以隔离关联策略；DanceTrack 官方 val 无 det.txt，混用会破坏口径一致。",
          "- 因此 DetA≈1，结论不能直接外推到含误检/漏检的真实检测器输出。",
          "- ReID 特征为 OSNet-x1_0 (MSMT17)，FP32，全策略共享同一缓存。",
          "- 跟踪器全程不接触 GT 身份；GT 仅用于事后分层与正确性核算。",
          "- 主指标为 TrackEval 官方口径；分带 ΔIDSW 为逐帧自实现统计，用于分层而非替代官方指标。", ""]
    txt = "\n".join(L)
    p = os.path.join(RESULTS, "difference_report.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
    print("DIFF_REPORT_WRITTEN", p, "reproduced=", reproduced, flush=True)


if __name__ == "__main__":
    main()
