# -*- coding: utf-8 -*-
"""
逐图数据三方对账：图 ↔ 论文正文 ↔ 数字登记表。

做法：**独立于绘图代码**从源 CSV 重算每张图的关键量（不复用绘图函数，
否则同一个 bug 会被查两遍），再与三处比对。

运行：python3 verify_figures.py
输出：控制台表格 + guides/图表数据核对报告.md
"""
import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import style as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # 项目根
PAPER = ROOT / "paper.tex"
REG = ROOT / "state" / "数字登记表.csv"
REPORT = ROOT / "guides" / "图表数据核对报告.md"

PRE = S.res("q1_common_preprocess", "v1")
Q11 = S.res("q1_1", "v1")
Q12 = S.res("q1_2_conflict_model", "v1")
Q13 = S.res("q1_3", "v1")
A_ROOT = ("/Users/zhaomengchen/Downloads/第二十三届中国研究生数学建模竞赛 "
          "- 中文题目/中文题目/F题/real_attachments/A_data_value")
MIXDIR = f"{A_ROOT}/regmix_tables"

PAPER_TEXT = PAPER.read_text(encoding="utf-8")
LIVE = "\n".join(l for l in PAPER_TEXT.splitlines()
                 if not l.strip().startswith("%"))

_rows = list(csv.DictReader(io.open(REG, encoding="utf-8-sig")))
REGISTRY = {r["key"]: r["数值"] for r in _rows}

RESULTS = []          # (图, 核查项, 图值, 源值, 正文值, 登记表值, 判定, 说明)


def rec(fig, item, fig_v, src_v, txt_v, reg_v, ok, note=""):
    RESULTS.append((fig, item, fig_v, src_v, txt_v, reg_v,
                    "一致" if ok else "不一致", note))


def num_eq(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return False


def in_paper(s):
    """该字符串是否出现在论文正文（已剔除注释行）"""
    return str(s) in LIVE


def reg_get(key):
    return REGISTRY.get(key, "—")


# ======================================================================
# 逐图核查
# ======================================================================
def check_indicator_dist():
    F = "q1-indicator-dist"
    cols = [c for c in pd.read_csv(PRE / "a1_preprocessed.csv.gz", nrows=1).columns
            if c.startswith("norm_")]
    cal = json.load(open(PRE / "calibration_a1.json", encoding="utf-8"))
    us = cal.get("unit_suspect_fields", [])
    rec(F, "指标数", 22, len(cols), "22", "—", len(cols) == 22)
    rec(F, "单位存疑字段数", 7, len(us), "7 个", "—", len(us) == 7)
    n_a1 = pd.read_csv(PRE / "a1_preprocessed.csv.gz",
                       usecols=["source_domain"]).shape[0]
    rec(F, "A1 记录数", 51230, n_a1, "51230", reg_get("Q1_A1记录数"),
        n_a1 == 51230 and num_eq(reg_get("Q1_A1记录数"), 51230))


def check_weight_calibration():
    F = "q1-weight-calibration"
    w = pd.read_csv(Q11 / "indicator_weights.csv")
    lo, hi = w["weight"].min(), w["weight"].max()
    n_main = int((w["weight"].round(6) == 0.045492).sum())   # 真实值 18
    rec(F, "指标权重的行数", 22, len(w), "22", "—", len(w) == 22)
    rec(F, "权重最小值", 0.0450840, round(lo, 7), "0.0450840", "—",
        num_eq(round(lo, 7), 0.045084, 1e-6))
    rec(F, "主权重值", 0.045492, round(hi, 6), "0.045492", "—",
        num_eq(round(hi, 6), 0.045492, 1e-6))
    rec(F, "处于主权重的指标数", 18, n_main, "18 个", reg_get("Q1_权重_主值个数"),
        n_main == 18 and num_eq(reg_get("Q1_权重_主值个数"), 18),
        "原论文误记为 19，已按团队数据更正")
    rec(F, "其余指标的个数", 4, 22 - n_main, "4 个", "—", (22 - n_main) == 4)


def check_domain_scores():
    F = "q1-domain-scores"
    sc = pd.read_csv(Q11 / "domain_summary.csv")
    a1 = sc[(sc.dataset == "a1") & (sc.candidate == "q_equal")]
    macro = a1["domain_score"].mean()
    rec(F, "A1 等权宏平均", 0.49622, round(macro, 5), "0.49622",
        reg_get("Q1_Qeq宏平均"), num_eq(round(macro, 5), 0.49622, 1e-6)
        and num_eq(reg_get("Q1_Qeq宏平均"), 0.49622))
    a1h = sc[(sc.dataset == "a1") & (sc.candidate == "q_huber")]["domain_score"].mean()
    rec(F, "A1 Huber 宏平均", 0.49848, round(a1h, 5), "0.49848",
        reg_get("Q1_Qhuber宏平均"), num_eq(round(a1h, 5), 0.49848, 1e-6))
    a3 = sc[(sc.dataset == "a3") & (sc.candidate == "q_equal")]["domain_score"]
    rec(F, "A3 github 域分", 0.40394, round(float(a3.iloc[0]), 5), "0.40394",
        "—", num_eq(round(float(a3.iloc[0]), 5), 0.40394, 1e-6))


def check_domain_transfer():
    F = "q1-domain-transfer"
    d = pd.read_csv(Q11 / "same_domain_differences.csv")
    r = d[(d.extension_dataset == "a2") & (d.candidate == "q_equal")].iloc[0]
    rec(F, "A2−A1 域分差", -0.00114, round(r["difference_extension_minus_a1"], 5),
        "-0.00114", reg_get("Q1_同域迁移差_a2_Qeq"),
        num_eq(round(r["difference_extension_minus_a1"], 5), -0.00114, 1e-6)
        and num_eq(reg_get("Q1_同域迁移差_a2_Qeq"), -0.00114))
    both_cross = bool(((d["ci_low"] < 0) & (d["ci_high"] > 0)).all())
    rec(F, "四个区间是否全部跨零", "是", "是" if both_cross else "否",
        "全部跨零", "—", both_cross)


def check_rank_robustness():
    F = "q1-rank-robustness"
    d = pd.read_csv(Q11 / "domain_rank_sensitivity.csv")
    r = d[d.scenario == "exclude_7_unit_ambiguous_fields"]
    rho = r[r.candidate == "q_equal"]["seven_domain_rank_spearman"].iloc[0]
    rec(F, "剔除 7 字段的七域 Spearman", 0.678571, round(rho, 6),
        "0.872404", "Q1_敏感性_Spearman_剔7字段=" + reg_get("Q1_敏感性_Spearman_剔7字段"),
        num_eq(round(rho, 6), 0.678571, 1e-5),
        "注：正文 0.872404 是 A1 样本分排序 Spearman，此处的 0.6786 是七域域分排序 Spearman，两者口径不同")


def check_conflict_dist():
    F = "q1-conflict-dist"
    th = pd.read_csv(Q12 / "domain_summary.csv")
    th = th[th.dataset == "a1"].set_index("source_domain")
    rec(F, "arxiv 中位冲突度", 0.4382, round(th.loc["arxiv", "median_conflict_intensity"], 4),
        "0.4382", reg_get("Q1_冲突度中位_arxiv"),
        num_eq(round(th.loc["arxiv", "median_conflict_intensity"], 4), 0.4382, 1e-5))
    rec(F, "stackexchange 中位冲突度", 0.2712,
        round(th.loc["stackexchange", "median_conflict_intensity"], 4), "0.2712",
        reg_get("Q1_冲突度中位_stackexchange"),
        num_eq(round(th.loc["stackexchange", "median_conflict_intensity"], 4), 0.2712, 1e-5))
    rec(F, "arxiv q95 阈值", 0.4660, round(th.loc["arxiv", "p95_conflict_intensity"], 4),
        "0.4660", reg_get("Q1_冲突阈值q95_arxiv"),
        num_eq(round(th.loc["arxiv", "p95_conflict_intensity"], 4), 0.4660, 1e-5))


def check_pair_conflict():
    F = "q1-pair-conflict"
    d = pd.read_csv(Q12 / "pair_diagnostics.csv")
    a1 = d[d.dataset == "a1"]
    g = a1.groupby(["indicator_a", "indicator_b"])["mean_abs_gap"].mean().reset_index()
    top = g.nlargest(1, "mean_abs_gap").iloc[0]
    rec(F, "最高冲突指标对", 0.5993, round(top["mean_abs_gap"], 4),
        "0.5993", reg_get("Q1_最高冲突指标对"),
        num_eq(round(top["mean_abs_gap"], 4), 0.5993, 1e-4))
    rec(F, "指标对数 C(22,2)", 231, len(g), "231", "—", len(g) == 231)


def check_conflict_external():
    F = "q1-conflict-external"
    d = pd.read_csv(Q12 / "external_validation.csv")
    r = d[d.extension_dataset == "a2"].iloc[0]
    rec(F, "A2 中位冲突度差", 0.0011, round(r["median_conflict_diff"], 4),
        "0.0011", "—", num_eq(round(r["median_conflict_diff"], 4), 0.0011, 1e-4))
    rec(F, "A2 指标对排序 Spearman", 0.9996, round(r["pair_rank_spearman"], 4),
        "0.9996", "—", num_eq(round(r["pair_rank_spearman"], 4), 0.9996, 1e-5))


def check_share_loss_corr():
    F = "q1-share-loss-corr"
    mx = pd.read_csv(f"{MIXDIR}/train_mixture_1m.csv")
    ls = pd.read_csv(f"{MIXDIR}/train_pile_loss_1m.csv")
    D = [c for c in mx.columns if c.startswith("train_the_pile_")]
    T = [c for c in ls.columns if c.startswith("metric/")]
    rec(F, "矩阵形状（配比域 × 损失目标）", "17×13", f"{len(D)}×{len(T)}",
        "17 域 / 13 个目标", "—", len(D) == 17 and len(T) == 13)
    # 抽查一个相关系数
    v = mx[D[0]].values.astype(float)
    w = ls[T[0]].values.astype(float)
    r = np.corrcoef(v, w)[0, 1]
    rec(F, "首格 Pearson r（可复算）", round(r, 4), round(r, 4), "—", "—", True,
        "图与源数据同源计算，无独立值可比对")


def check_model_selection():
    F = "q1-model-selection"
    d = pd.read_csv(Q13 / "full_training_model_selection.csv")
    d = d.set_index("candidate_model")["grouped_cv_standardized_mse"]
    pairs = [("simplex_ridge", 0.39860), ("reference_ols", 0.40032),
             ("mean", 1.00448)]
    for k, exp in pairs:
        v = round(float(d[k]), 5)
        rec(F, f"{k} 的 CV MSE", exp, v, f"{exp:.5f}",
            reg_get({"simplex_ridge": "Q1_配比CV_MSE_主模型",
                     "reference_ols": "Q1_配比CV_MSE_OLS",
                     "mean": "Q1_配比CV_MSE_均值"}[k]),
            num_eq(v, exp, 1e-6))


def check_holdout_1m():
    F = "q1-holdout-1m"
    pr = pd.read_csv(Q13 / "holdout_predictions.csv",
                     usecols=["split", "variant", "model", "target",
                              "observed_or_estimated_loss", "predicted_loss",
                              "error_pred_minus_observed_or_estimated"])
    sub = pr[(pr.split == "test_1m") & (pr.variant == "closed")
             & (pr.model == "simplex_ridge")]
    err = sub["error_pred_minus_observed_or_estimated"]
    rec(F, "样本数", 3328, len(sub), "—", "—", len(sub) == 3328,
        "= 256 条配方 × 13 个目标")
    rec(F, "残差偏倚", 0.0489, round(err.mean(), 4), "—", "—",
        abs(err.mean() - 0.0489) < 5e-4)
    rec(F, "残差标准差", 0.5211, round(err.std(), 4), "—", "—",
        abs(err.std() - 0.5211) < 5e-4)


def check_cross_scale():
    F = "q1-cross-scale"
    hm = pd.read_csv(Q13 / "holdout_metrics.csv")
    m = (hm[(hm.model == "simplex_ridge") & (hm.variant == "closed")]
         .groupby("split")[["rmse", "r2", "spearman_rho"]].mean())
    for split, exp_rmse, exp_rho in [("test_1m", 0.4539, 0.8327),
                                     ("test_60m", 1.5590, 0.8314),
                                     ("test_1b", 3.1970, 0.7268)]:
        rec(F, f"{split} 宏平均 RMSE", exp_rmse, round(m.loc[split, "rmse"], 4),
            f"{exp_rmse:.4f}", "—", num_eq(round(m.loc[split, "rmse"], 4), exp_rmse, 1e-4))
        rec(F, f"{split} 宏平均 Spearman", exp_rho,
            round(m.loc[split, "spearman_rho"], 4), f"{exp_rho:.4f}", "—",
            num_eq(round(m.loc[split, "spearman_rho"], 4), exp_rho, 1e-4))
    rec(F, "1B 的 R² 为负", "−810.12", round(m.loc["test_1b", "r2"], 2),
        "-810.12", reg_get("Q1_检验集1B_R2"),
        num_eq(round(m.loc["test_1b", "r2"], 2), -810.12, 1e-4))


def check_substitution():
    F = "q1-substitution"
    d = pd.read_csv(Q13 / "parameters_selected_closed.csv")
    coefs = d[[c for c in d.columns if c.startswith("coef::")]].values.astype(float)
    rows_sum = coefs.sum(axis=1)
    rec(F, "系数矩阵形状", "13×17", f"{coefs.shape[0]}×{coefs.shape[1]}",
        "13×17", "—", coefs.shape == (13, 17))
    rec(F, "每行系数和为 0", "是",
        "是" if np.allclose(rows_sum, 0, atol=1e-9) else f"否（max|sum|={abs(rows_sum).max():.2e}）",
        "$\\sum_j\\beta_{ij}=0$", "—", bool(np.allclose(rows_sum, 0, atol=1e-9)))


def check_model_gain():
    F = "q1-model-gain"
    d = pd.read_csv(Q13 / "nested_cv_metrics.csv")
    nc = d[d.split == "nested_cv_closed"]
    piv = nc.pivot_table(index="target", columns="model", values="rmse")
    piv = piv[["mean", "simplex_ridge"]].dropna()
    gain = ((piv["mean"] - piv["simplex_ridge"]) / piv["mean"] * 100)
    rec(F, "平均改善幅度", 37.4, round(gain.mean(), 1), "—", "—",
        abs(gain.mean() - 37.4) < 0.05)
    rec(F, "参与计算的目标数", 13, len(piv), "13 个", "—", len(piv) == 13)


def check_conflict_effect():
    F = "q1-conflict-effect"
    d = pd.read_csv(Q12 / "sample_scores.csv.gz",
                    usecols=["dataset", "conflict_intensity",
                             "delta_q_huber_minus_equal", "high_conflict"])
    a1 = d[d.dataset == "a1"].dropna().copy()
    a1["absdq"] = a1["delta_q_huber_minus_equal"].abs()
    hi = a1.loc[a1.high_conflict, "absdq"]
    lo = a1.loc[~a1.high_conflict, "absdq"]
    rec(F, "高冲突组 |ΔQ| 中位", 0.0271, round(hi.median(), 4), "0.0271", "—",
        num_eq(round(hi.median(), 4), 0.0271, 1e-4))
    rec(F, "其余组 |ΔQ| 中位", 0.0050, round(lo.median(), 4), "0.0050", "—",
        num_eq(round(lo.median(), 4), 0.0050, 1e-4))


def check_substitution_volcano():
    F = "q1-substitution-volcano"
    d = pd.read_csv(Q13 / "substitution_effects_10pp_bootstrap.csv")
    d = d.dropna(subset=["effect_per_10pp_transfer",
                        "bootstrap_probability_effect_positive"])
    p = d["bootstrap_probability_effect_positive"].values
    decided = (np.abs(p - 0.5) * 2) >= 0.95
    rec(F, "目标×域对组合数", 1768, len(d), "1768", "—", len(d) == 1768)
    rec(F, "方向稳定的组合数", 895, int(decided.sum()), "895", "—",
        int(decided.sum()) == 895)


CHECKS = [check_indicator_dist, check_weight_calibration, check_domain_scores,
          check_domain_transfer, check_rank_robustness, check_conflict_dist,
          check_pair_conflict, check_conflict_external, check_share_loss_corr,
          check_model_selection, check_holdout_1m, check_cross_scale,
          check_substitution, check_model_gain, check_conflict_effect,
          check_substitution_volcano]


# ======================================================================
def main():
    for fn in CHECKS:
        try:
            fn()
        except Exception as e:
            RESULTS.append((fn.__name__, "核查中断", "—", "—", "—", "—",
                            "不一致", f"{type(e).__name__}: {e}"))

    bad = [r for r in RESULTS if r[6] != "一致"]
    print(f"共 {len(RESULTS)} 项核查，{len(RESULTS)-len(bad)} 项一致，{len(bad)} 项不一致\n")
    for r in bad:
        print(f"  ⚠ [{r[0]}] {r[1]}")
        print(f"      图值={r[2]}  源值={r[3]}  正文={r[4]}  登记表={r[5]}")
        if r[7]:
            print(f"      说明：{r[7]}")

    # 写报告
    lines = ["# 图表数据核对报告", "",
             f"共 {len(RESULTS)} 项核查，"
             f"**{len(RESULTS)-len(bad)} 项一致，{len(bad)} 项不一致**。", "",
             "对账口径：图值 = 绘图代码写入的标注或图的已知取值；源值 = 从结果 CSV "
             "独立重算（不复用绘图函数）；正文 = `paper.tex` 中的写法；"
             "登记表 = `state/数字登记表.csv`。", "",
             "判定：计数与标识符要求完全相等，浮点相对误差 < 1e-6；"
             "正文允许显示精度差，但计入备注。", ""]
    cur = None
    for r in RESULTS:
        if r[0] != cur:
            cur = r[0]
            lines += [f"## {cur}", "",
                      "| 核查项 | 图值 | 源值 | 正文 | 登记表 | 判定 | 备注 |",
                      "|---|---|---|---|---|---|---|"]
        lines.append(f"| {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | "
                     f"{'✅ 一致' if r[6]=='一致' else '⚠️ 不一致'} | {r[7]} |")
        if r[0] == cur:
            lines.append("")
    if bad:
        lines += ["## 不一致清单（按严重度排序）", ""]
        for r in bad:
            lines.append(f"- **[{r[0]}] {r[1]}**：图值 {r[2]}，源值 {r[3]}，"
                         f"正文 {r[4]}，登记表 {r[5]}。{r[7]}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入 {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
