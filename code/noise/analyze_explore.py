#!/usr/bin/env python3
"""Explore-stage analysis of the frozen noise-gradient cells.

PRE-REGISTERED / MODELLING ONLY (design sec.4.4):
  * read the 5040 explore cells (never regenerate, never filter by outcome),
  * compute DeltaAssA = AssA(A4) - AssA(A0) and the pre-declared secondary
    readouts (DeltaIDSW, DeltaDetA, IDF1),
  * cluster-bootstrap 95% CI per frozen condition (1000 resamples, seed fixed),
  * locate the three boundary quantities eta*_0 / eta_dagger / eta_flip,
  * fit the <=8-feature label-free logistic criterion P and FREEZE it,
  * run leave-one-sequence-out diagnostics (modelling diagnostics ONLY;
    the formal H2 verdict belongs to the confirm stage).

NO verdict on H1/H2/H3 is issued here. No threshold, condition, feature or
dataset is added, removed or retuned on the basis of these numbers.
"""
from __future__ import annotations

import os
import json
import glob
import math
import hashlib
import argparse
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CELLS = "/path/to/mot_exp_backup/mirror/exp_noise/results/cells_explore"
OBSERVABLES = ("mean_score", "p10_score", "score_gap", "mean_pair_iou",
               "det_per_frame", "scale_cv", "common_mode_energy",
               "low_score_frac")
MAIN_RESPONSE = ("A4", "A0")
L2 = 1.0                 # fixed before seeing any result
THRESHOLD = 0.5          # fixed before seeing any result
N_BOOT = 1000
BOOT_SEED = 20261115


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_obj(o):
    s = json.dumps(o, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_cells(cells_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(cells_dir, "*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def integrity(rows, cfg):
    """Hard gate: refuse to analyse an incomplete or mixed-freeze result set."""
    checks = []

    def add(name, ok, detail):
        checks.append(dict(name=name, passed=bool(ok), detail=str(detail)))

    add("n_cells", len(rows) == 5040, "%d rows (expected 5040)" % len(rows))
    digests = {r["frozen_sha256"] for r in rows}
    add("single_freeze", len(digests) == 1 and
        digests == {cfg["integrity"]["digest"]}, sorted(digests)[:2])
    add("stage", {r["stage"] for r in rows} == {"explore"},
        sorted({r["stage"] for r in rows}))
    add("seeds", {r["seed"] for r in rows} == {11, 23, 37},
        sorted({r["seed"] for r in rows}))
    add("strategies", {r["strategy"] for r in rows} ==
        {"A0", "A1", "A2", "A3", "A4", "A5"},
        sorted({r["strategy"] for r in rows}))
    add("conditions", {r["cond"] for r in rows} == set(range(28)),
        sorted({r["cond"] for r in rows}))
    add("benchmarks", {r["benchmark"] for r in rows} ==
        {"mot17", "dancetrack"},
        sorted({r["benchmark"] for r in rows}))
    seqs = {(r["benchmark"], r["seq"]) for r in rows}
    add("n_sequences", len(seqs) == 10, "%d sequences" % len(seqs))
    grp = defaultdict(set)
    for r in rows:
        grp[(r["benchmark"], r["seq"], r["cond"], r["seed"])].add(r["strategy"])
    add("n_groups", len(grp) == 840, "%d groups" % len(grp))
    add("groups_complete", all(len(v) == 6 for v in grp.values()),
        "%d incomplete" % sum(1 for v in grp.values() if len(v) != 6))
    add("observables_complete",
        all(tuple(r["observables"]) == OBSERVABLES for r in rows),
        "8 label-free features on every cell")
    return checks


def build_groups(rows):
    """One analysis unit = (benchmark, seq, cond, seed)."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["benchmark"], r["seq"], r["cond"], r["seed"])][r["strategy"]] = r
    out = []
    for key, d in sorted(by.items()):
        a4, a0 = d["A4"], d["A0"]
        o = dict(benchmark=key[0], seq=key[1], cond=key[2], seed=key[3],
                 theta=a4["theta"],
                 dAssA=float(a4["AssA"] - a0["AssA"]),
                 dIDSW=float(a4["IDSW"] - a0["IDSW"]),
                 dDetA=float(a4["DetA"] - a0["DetA"]),
                 dIDF1=float(a4["IDF1"] - a0["IDF1"]),
                 detA_A0=float(a0["DetA"]),
                 assA_A0=float(a0["AssA"]),
                 assA_A4=float(a4["AssA"]),
                 reid_calls=float(a4["reid_calls"]),
                 n_abstain=float(a4["n_abstain"]),
                 n_amb=float(a4["n_amb"]),
                 observables={k: float(a4["observables"][k])
                              for k in OBSERVABLES})
        out.append(o)
    return out


def cluster_bootstrap_ci(vals, seq_labels, n=N_BOOT, seed=BOOT_SEED):
    """Resample SEQUENCES (all seeds of a sequence travel together)."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for lab, v in zip(seq_labels, vals):
        groups[lab].append(v)
    labs = sorted(groups)
    arrs = [np.asarray(groups[k], dtype=np.float64) for k in labs]
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        pick = rng.integers(0, len(labs), len(labs))
        chunk = np.concatenate([arrs[j] for j in pick])
        means[i] = chunk.mean()
    return dict(mean=float(np.mean(vals)),
                ci_lo=float(np.quantile(means, 0.025)),
                ci_hi=float(np.quantile(means, 0.975)),
                n_resamples=n)


def condition_table(groups):
    """Per frozen condition: bootstrapped main response + secondary readouts."""
    by = defaultdict(list)
    for g in groups:
        by[g["cond"]].append(g)
    out = {}
    for c, gs in sorted(by.items()):
        labs = ["%s/%s" % (g["benchmark"], g["seq"]) for g in gs]
        out[str(c)] = dict(
            theta=gs[0]["theta"],
            n_units=len(gs),
            dAssA=cluster_bootstrap_ci([g["dAssA"] for g in gs], labs),
            dIDSW=cluster_bootstrap_ci([g["dIDSW"] for g in gs], labs),
            dIDF1=cluster_bootstrap_ci([g["dIDF1"] for g in gs], labs),
            detA_A0=float(np.mean([g["detA_A0"] for g in gs])),
            reid_calls=float(np.mean([g["reid_calls"] for g in gs])),
            n_abstain=float(np.mean([g["n_abstain"] for g in gs])),
            n_amb=float(np.mean([g["n_amb"] for g in gs])),
        )
    return out


def families(cfg):
    """Classify the 28 FROZEN conditions without looking at any outcome."""
    conds = cfg["conditions"]
    fam = {}
    for c, t in enumerate(conds):
        th = t["theta"] if "theta" in t else t
        s, b, l, tau = (float(th["sigma_loc"]), float(th["beta"]),
                        float(th["lambd"]), float(th["tau"]))
        nz = sum(1 for v in (s, b, l) if v > 0)
        # A single-component sweep means EXACTLY ONE component differs from
        # its baseline value. tau != 1 is itself a deviation, so any condition
        # with tau != 1 AND a non-zero continuous component is an interaction.
        # (First version of this function forgot the tau==1 guard and leaked
        #  the two beta x tau interaction cells into the beta_only curve,
        #  producing duplicate eta points and a spurious interior extremum.)
        if nz == 0 and tau == 1:
            fam[c] = "baseline"
        elif nz == 0:
            fam[c] = "tau_only"
        elif nz == 1 and tau == 1:
            fam[c] = ("sigma_only" if s > 0 else
                      "beta_only" if b > 0 else "lambda_only")
        else:
            fam[c] = "interaction"
    return fam


def _interp_zero(xs, ys):
    """First sign crossing (positive -> non-positive), linearly interpolated."""
    for i in range(1, len(xs)):
        if ys[i - 1] > 0 >= ys[i]:
            y0, y1 = ys[i - 1], ys[i]
            if y0 == y1:
                return float(xs[i])
            return float(xs[i - 1] + (xs[i] - xs[i - 1]) * y0 / (y0 - y1))
    return None


def _deriv_zero(xs, ys):
    """First interior extremum: d(DeltaAssA)/d(eta)=0, as frozen in
    frozen_config.response.boundaries.eta_star_0. Discrete slopes are
    linearly interpolated to locate the zero of the derivative."""
    slopes = []
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        if dx <= 0:
            continue
        slopes.append(((ys[i] - ys[i - 1]) / dx,
                       0.5 * (xs[i] + xs[i - 1])))
    for j in range(1, len(slopes)):
        s0, x0 = slopes[j - 1]
        s1, x1 = slopes[j]
        if s0 * s1 < 0:
            return float(x0 - s0 * (x1 - x0) / (s1 - s0))
    return None


def _first_above(xs, ys, target=0.0):
    for i in range(len(xs)):
        if ys[i] > target:
            if i == 0:
                return float(xs[0])
            y0, y1 = ys[i - 1], ys[i]
            if y1 == y0:
                return float(xs[i])
            return float(xs[i - 1] + (xs[i] - xs[i - 1]) *
                         (target - y0) / (y1 - y0))
    return None


def boundaries(cfg, ctab, fam):
    """Three boundary quantities per single-component family."""
    key_of = dict(sigma_only="sigma_loc", beta_only="beta",
                  lambda_only="lambd", tau_only="tau")
    base_cond = [c for c, f in fam.items() if f == "baseline"]
    base_detA = (ctab[str(base_cond[0])]["detA_A0"]
                 if len(base_cond) == 1 else None)
    out = {}
    for f, k in key_of.items():
        pts = [(float(ctab[str(c)]["theta"][k]), c)
               for c, ff in fam.items() if ff == f]
        # The zero level is shared by all single-component sweeps (frozen
        # condition 0), so every curve must start at eta=0 for the boundary
        # interpolation to be well-defined. tau's own baseline value is 1.
        if base_cond:
            v0 = (float(ctab[str(base_cond[0])]["theta"][k])
                  if f == "tau_only" else 0.0)
            pts.append((v0, base_cond[0]))
        pts.sort()
        xs = [p[0] for p in pts]
        dassa = [ctab[str(c)]["dAssA"]["mean"] for _, c in pts]
        didsw = [ctab[str(c)]["dIDSW"]["mean"] for _, c in pts]
        deta = [ctab[str(c)]["detA_A0"] for _, c in pts]
        # eta_dagger: |DeltaAssA| smaller than the detection-quality loss
        # relative to the no-noise baseline (design sec.3.3)
        det_loss = ([abs(base_detA - v) for v in deta]
                    if base_detA is not None else [0.0] * len(deta))
        swallowed = [abs(dassa[i]) < det_loss[i] for i in range(len(xs))]
        first_swallow = next((xs[i] for i, v in enumerate(swallowed)
                              if v and xs[i] > 0), None)
        # non-monotonicity: a strict interior extremum
        interior_extremum = any(
            (dassa[i] > dassa[i - 1] and dassa[i] > dassa[i + 1]) or
            (dassa[i] < dassa[i - 1] and dassa[i] < dassa[i + 1])
            for i in range(1, len(dassa) - 1))
        out[f] = dict(
            component=k, xs=xs, dAssA=dassa, dIDSW=didsw,
            detA_A0=deta, det_loss=det_loss,
            eta_star_0=_deriv_zero(xs, dassa),
            eta_star_0_zero_crossing=_interp_zero(xs, dassa),
            eta_flip=_first_above(xs, didsw, 0.0),
            eta_dagger=first_swallow,
            interior_extremum=bool(interior_extremum),
            sign_any_positive=any(v > 0 for v in dassa),
            sign_any_negative=any(v < 0 for v in dassa),
            dAssA_range=[float(min(dassa)), float(max(dassa))],
        )
    return out


class Logistic:
    """Pure-numpy L2 logistic regression (no sklearn on the server)."""

    def __init__(self, l2=L2):
        self.l2 = float(l2)

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def fit(self, X, y):
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Z = (X - mu) / sd
        n, d = Z.shape
        A = np.hstack([Z, np.ones((n, 1))])
        beta = np.zeros(d + 1)
        reg = np.eye(d + 1) * self.l2
        reg[-1, -1] = 0.0            # never penalise the intercept
        for _ in range(120):
            p = self._sig(A @ beta)
            w = np.clip(p * (1 - p), 1e-8, None)
            g = A.T @ (y - p) - reg @ beta
            H = A.T @ (w[:, None] * A) + reg
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            beta += step
            if np.max(np.abs(step)) < 1e-9:
                break
        self.mu, self.sd, self.beta = mu, sd, beta
        self.zero_var_ = X.std(axis=0) < 1e-12
        # coefficients on the ORIGINAL feature scale; zero-variance columns
        # are exactly zero and are reported as degenerate
        self.coef_ = beta[:-1] / sd
        self.coef_[self.zero_var_] = 0.0
        self.intercept_ = float(beta[-1] - np.sum(self.coef_ * mu))
        return self

    def decision(self, X):
        return (X - self.mu) / self.sd @ self.beta[:-1] + self.beta[-1]

    def predict(self, X, threshold=THRESHOLD):
        return (self._sig(self.decision(X)) >= threshold).astype(int)

    def to_json(self):
        return dict(model="logistic_regression_l2", l2=self.l2,
                    threshold=THRESHOLD, features=list(OBSERVABLES),
                    scaler_mean=[float(v) for v in self.mu],
                    scaler_std=[float(v) for v in self.sd],
                    beta_std=[float(v) for v in self.beta],
                    coef_original_scale=[float(v) for v in self.coef_],
                    intercept=float(self.intercept_),
                    zero_variance_features=[OBSERVABLES[i] for i, z
                                           in enumerate(self.zero_var_) if z])


def design_matrix(units, cols):
    return np.asarray([[u["observables"][k] for k in cols]
                       for u in units], dtype=np.float64)


def loo_diagnostics(units):
    """Modelling diagnostics only. NOT the formal H2 test (confirm stage)."""
    labels = ["%s/%s" % (u["benchmark"], u["seq"]) for u in units]
    y = np.asarray([1 if u["dAssA"] > 0 else 0 for u in units],
                   dtype=np.float64)
    X = design_matrix(units, OBSERVABLES)
    specs = {
        "P_full_8": OBSERVABLES,
        "baseline_confidence": ("mean_score", "p10_score"),
        "baseline_beta_lambda_proxy": ("det_per_frame", "low_score_frac"),
    }
    folds = sorted(set(labels))
    out = {}
    for name, cols in specs.items():
        Xi = design_matrix(units, cols)
        pred = np.zeros(len(units), dtype=int)
        for f in folds:
            te = np.asarray([i for i, l in enumerate(labels) if l == f])
            tr = np.asarray([i for i, l in enumerate(labels) if l != f])
            m = Logistic().fit(Xi[tr], y[tr])
            pred[te] = m.predict(Xi[te])
        out[name] = dict(features=list(cols),
                         accuracy=float((pred == y).mean()),
                         n=len(units), n_folds=len(folds))
    maj = int(y.mean() >= 0.5)
    out["constant_majority"] = dict(
        features=[], accuracy=float((np.full(len(y), maj) == y).mean()),
        n=len(units), n_folds=len(folds), majority_class=maj)
    # uncertainty of P's LOO accuracy, resampling held-out units
    rng = np.random.default_rng(BOOT_SEED)
    acc = np.asarray([1 if u["dAssA"] > 0 else 0 for u in units])
    Xi = X
    pred = np.zeros(len(units), dtype=int)
    for f in folds:
        te = np.asarray([i for i, l in enumerate(labels) if l == f])
        tr = np.asarray([i for i, l in enumerate(labels) if l != f])
        m = Logistic().fit(Xi[tr], y[tr])
        pred[te] = m.predict(Xi[te])
    hit = (pred == acc).astype(float)
    boots = np.array([hit[rng.integers(0, len(hit), len(hit))].mean()
                      for _ in range(N_BOOT)])
    out["P_full_8"]["ci_lo"] = float(np.quantile(boots, 0.025))
    out["P_full_8"]["ci_hi"] = float(np.quantile(boots, 0.975))
    out["note"] = ("leave-one-SEQUENCE-out inside the explore subset; "
                   "modelling diagnostic only, NOT the formal H2 verdict")
    return out, Logistic().fit(X, y)


def markdown(report):
    L = []
    A = L.append
    A("# Phase 2 探索子集 · 汇总报告（建模用，非正式判定）")
    A("")
    A("- 生成时间：%s" % report["utc"])
    A("- 冻结配置：`%s`（r%s）" % (report["frozen_sha256"],
                                 report["freeze_revision"]))
    A("- 数据：%d cells / %d 组 / %d 序列，全部完整性与 hash 校验通过" %
      (report["n_cells"], report["n_groups"], report["n_sequences"]))
    A("- **按设计 §4.4，本阶段只用于拟合判据 P、估计边界粗位置；"
      "H1/H2/H3 的正式判定在确认阶段一次性给出。**")
    A("")
    A("## 1. 响应量总览")
    A("")
    v = report["overall"]
    A("| 量 | 均值 | 95% CI |")
    A("|---|---|---|")
    for k, lab in (("dAssA", "ΔAssA (A4−A0)"),
                   ("dIDSW", "ΔIDSW (负为好)"),
                   ("dIDF1", "ΔIDF1")):
        A("| %s | %+.6f | [%+.6f, %+.6f] |" %
          (lab, v[k]["mean"], v[k]["ci_lo"], v[k]["ci_hi"]))
    A("")
    A("正 ΔAssA 组占比：%.1f%%；负占比 %.1f%%" %
      (100 * report["sign_fraction_positive"],
       100 * (1 - report["sign_fraction_positive"])))
    A("")
    A("## 2. 按噪声分量的边际收益（各档 ΔAssA 均值）")
    A("")
    for f, b in report["boundaries"].items():
        A("### %s（分量 `%s`）" % (f, b["component"]))
        A("")
        A("| η | ΔAssA | ΔIDSW | DetA(A0) |")
        A("|---|---|---|---|")
        for x, d, s, da in zip(b["xs"], b["dAssA"], b["dIDSW"],
                               b["detA_A0"]):
            A("| %.6f | %+.6f | %+.1f | %.4f |" % (x, d, s, da))
        A("")
        A("- `η*_0`（冻结定义 `∂ΔAssA/∂η=0` 的首个内部极值）= %s；"
          "ΔAssA 首次变号点 = %s" %
          (b["eta_star_0"], b["eta_star_0_zero_crossing"]))
        A("- `η_flip`（ΔIDSW 转正）= %s" % b["eta_flip"])
        A("- `η†`（收益被检测退化吞没）= %s" % b["eta_dagger"])
        A("- 存在内部极值：%s；ΔAssA 范围 [%+.6f, %+.6f]" %
          (b["interior_extremum"], b["dAssA_range"][0],
           b["dAssA_range"][1]))
        A("")
    A("## 3. 判据 P（已冻结）")
    A("")
    p = report["criterion"]
    A("- 模型：L2 逻辑回归，%d 个无标签观测量，L2=%s，阈值=%s" %
      (len(p["features"]), p["l2"], p["threshold"]))
    A("- 系数（原始尺度）：")
    A("")
    A("| 特征 | 系数 |")
    A("|---|---|")
    for k, c in zip(p["features"], p["coef_original_scale"]):
        A("| `%s` | %+.6f |" % (k, c))
    A("")
    A("- 截距：%+.6f；判据 sha256：`%s`" %
      (p["intercept"], report["criterion_sha256"]))
    A("")
    A("## 4. 留一序列诊断（**非 H2 判定**）")
    A("")
    A("| 模型 | 特征数 | 符号准确率 | 95% CI |")
    A("|---|---|---|---|")
    for name, d in report["loo_diagnostics"].items():
        if not isinstance(d, dict):
            continue
        ci = ("[%s, %s]" % (("%.4f" % d["ci_lo"]), ("%.4f" % d["ci_hi"]))
              if "ci_lo" in d else "—")
        A("| %s | %d | %.4f | %s |" %
          (name, len(d["features"]), d["accuracy"], ci))
    A("")
    A("> 这是探索子集内部的拟合诊断，用来判断 P 是否值得冻结；"
      "H2 的成立与否必须在确认阶段、以三种留一验证 + Holm 校正给出。")
    A("")
    A("## 5. 口径与限制（如实记录）")
    A("")
    A("- `AssA` 用 TrackEval 官方 alignment-weighted 匹配移植，"
      "本机自检与官方 `HOTA.eval_sequence` 差值 0；但 TP 集合可随 "
      "tracker id 分配合法变化，各策略 DetA 因此有微小差异。")
    A("- `score_gap` 在该置信度池下恒为常数，逻辑回归自动给零系数；"
      "未为它改动注入器。")
    A("- `η*_0` 按冻结定义取 `∂ΔAssA/∂η=0`（离散斜率变号、线性插值）；"
      "同时报告 ΔAssA 首次变号点，两者都给、不做择优。")
    A("- `η†` 的可操作化定义为 `|ΔAssA| < |DetA(η) − DetA(0)|` 的最小 η"
      "（检测退化以无噪声 baseline 的 A0.DetA 为参照）。")
    A("- 官方 det 锚点口径（外部效度）留到确认阶段与注入口径同时报告。")
    A("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(HERE, "frozen_config.json"),
                         encoding="utf-8"))
    rows = load_cells(a.cells)
    checks = integrity(rows, cfg)
    failed = [c for c in checks if not c["passed"]]
    print("=== INTEGRITY: %d/%d passed ===" %
          (len(checks) - len(failed), len(checks)))
    for c in failed:
        print("  FAILED", c["name"], c["detail"])
    if failed:
        raise SystemExit("INTEGRITY_GATE_FAILED - refusing to analyse")

    groups = build_groups(rows)
    ctab = condition_table(groups)
    fam = families(cfg)
    bnd = boundaries(cfg, ctab, fam)
    loo, fitted = loo_diagnostics(groups)

    overall = {}
    for k in ("dAssA", "dIDSW", "dIDF1"):
        labs = ["%s/%s" % (g["benchmark"], g["seq"]) for g in groups]
        overall[k] = cluster_bootstrap_ci([g[k] for g in groups], labs)

    criterion = fitted.to_json()
    seal = dict(model=criterion,
                source_stage="explore",
                frozen_sha256=cfg["integrity"]["digest"],
                n_fit_units=len(groups),
                fit_utc=utc(),
                note="FROZEN ONCE after explore; confirm stage may only "
                     "APPLY these parameters, never refit or retune")
    # digest covers model parameters ONLY: including fit_utc would make the
    # seal change on every re-run even though the fitted model is identical.
    seal["digest"] = sha256_obj(dict(model=criterion,
                                     source_stage=seal["source_stage"],
                                     frozen_sha256=seal["frozen_sha256"],
                                     n_fit_units=seal["n_fit_units"],
                                     note=seal["note"]))

    report = dict(
        utc=utc(), stage="explore", role="MODELLING_ONLY_NO_VERDICT",
        frozen_sha256=cfg["integrity"]["digest"],
        freeze_revision=cfg["freeze_revision"],
        n_cells=len(rows), n_groups=len(groups),
        n_sequences=len({(r["benchmark"], r["seq"]) for r in rows}),
        integrity_checks=checks,
        overall=overall,
        sign_fraction_positive=float(np.mean(
            [g["dAssA"] > 0 for g in groups])),
        conditions=ctab,
        family_counts={f: sum(1 for x in fam.values() if x == f)
                       for f in sorted(set(fam.values()))},
        boundaries=bnd,
        loo_diagnostics=loo,
        criterion=criterion,
        criterion_sha256=seal["digest"],
        statistics=dict(bootstrap_n=N_BOOT, bootstrap_seed=BOOT_SEED,
                        ci="0.95", resample_unit="sequence (all seeds)"),
    )
    with open(os.path.join(a.out, "explore_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(a.out, "frozen_criterion.json"), "w",
              encoding="utf-8") as f:
        json.dump(seal, f, indent=2, ensure_ascii=False)
    with open(os.path.join(a.out, "EXPLORE_REPORT.md"), "w",
              encoding="utf-8") as f:
        f.write(markdown(report))

    print("=== OVERALL ===")
    for k in ("dAssA", "dIDSW", "dIDF1"):
        v = overall[k]
        print("  %-6s mean=%+.6f CI=[%+.6f, %+.6f]" %
              (k, v["mean"], v["ci_lo"], v["ci_hi"]))
    print("  sign fraction positive = %.3f" %
          report["sign_fraction_positive"])
    print("=== FAMILY BOUNDARIES ===")
    for f, b in bnd.items():
        print("  %-12s eta*_0=%-10s eta_flip=%-10s eta_dag=%-10s "
              "range=[%+.4f,%+.4f] interior_ext=%s" %
              (f, b["eta_star_0"], b["eta_flip"], b["eta_dagger"],
               b["dAssA_range"][0], b["dAssA_range"][1],
               b["interior_extremum"]))
    print("=== LOO DIAGNOSTICS (not a verdict) ===")
    for name, d in loo.items():
        if isinstance(d, dict):
            print("  %-32s acc=%.4f" % (name, d["accuracy"]))
    print("=== CRITERION FROZEN sha256=%s ===" % seal["digest"])


if __name__ == "__main__":
    main()
